"""
Stage 5: 自动过滤 & 质量保证
============================
Phase 3 改进 — 借鉴 EGOMEMREASON 的自动过滤流程，
并加入 EgoMind 独有的 Sparsity 驱动校验。

三步过滤：
  1. 文本泄漏盲测 — 不给证据，让 LLM 裸答，答对说明题目太简单/泄题
  2. 时间跨度检查 — 证据 clips 必须跨越 ≥ MIN_EVIDENCE_SPAN_HOURS
  3. Sparsity 校验 — σ(q) 在合理范围内，避免"不需要搜索"的题

过滤后可选择自动重生成（Quality Loop）。

成本估算（纯文本 LLM）：
  盲测: N 道题 × M 个模型次 ≈ 15 × 1 = 15 次调用
  约 $0.01 总计

使用方式：
  python stage8_filter.py                    # 过滤 qa_pairs.json
  python stage8_filter.py --input qa_pairs.json  # 过滤指定文件
  python stage8_filter.py --dry-run          # 只看不跑
"""

import json
import os
import time
from collections import defaultdict
from openai import OpenAI
import config
from prompts import STAGE5_BLIND_SYSTEM, STAGE5_BLIND_USER


# ================================================================
# 过滤 1: 文本泄漏盲测
# ================================================================

def blind_test_question(client, model_name, question):
    """对单道题做盲测。返回是否答对。"""
    q_text = question.get("question", "")
    options = question.get("options", [])
    gt = question.get("ground_truth", "")

    options_line = f"OPTIONS: {options}" if options else ""
    user_msg = STAGE5_BLIND_USER.format(q_text=q_text, options_line=options_line)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": STAGE5_BLIND_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content.strip())
        answer = result.get("answer", "").strip().upper()
        gt_clean = str(gt).strip().upper()
        return answer == gt_clean or gt_clean in answer
    except Exception as e:
        print(f"      盲测异常: {e}")
        return False


def run_blind_tests(questions):
    """对全部 MCQ 题进行文本泄漏盲测。"""
    mcq_questions = [q for q in questions if q.get("type") == "multiple_choice"]
    if not mcq_questions:
        print("  无多选题，跳过盲测")
        return questions, []

    client = OpenAI(
        api_key=config.TEXT_LLM_API_KEY,
        base_url=config.TEXT_LLM_BASE_URL if config.TEXT_LLM_BASE_URL else None
    )

    leaked = []
    passed = []

    for q in questions:
        if q.get("type") != "multiple_choice":
            passed.append(q)
            continue

        qid = q.get("id", "?")
        print(f"    盲测 {qid}...", end=" ")

        correct_count = 0
        for model_name in config.BLIND_TEST_MODELS:
            if blind_test_question(client, model_name, q):
                correct_count += 1

        ratio = correct_count / max(len(config.BLIND_TEST_MODELS), 1)
        if ratio >= config.BLIND_TEST_THRESHOLD:
            print(f"✗ 泄漏 ({correct_count}/{len(config.BLIND_TEST_MODELS)} 答对)")
            leaked.append(q)
        else:
            print(f"✓ ({correct_count}/{len(config.BLIND_TEST_MODELS)} 答对)")
            passed.append(q)

    return passed, leaked


# ================================================================
# 过滤 2: 时间跨度检查
# ================================================================

def parse_timestamp(ts_str):
    """解析 HH:MM:SS 时间戳为秒数。"""
    parts = ts_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    return 0


def check_temporal_span(question, captions):
    """检查问题的 evidence clips 时间跨度是否满足最小要求。"""
    statements = question.get("evidence_statements", [])
    if not statements:
        # 没有 statements，回退到 source_evidence
        return True  # 无法判断，放行

    timestamps = []
    for s in statements:
        clip_id = s.get("clip_id", "")
        if clip_id in captions:
            ts = captions[clip_id].get("timestamp", "")
            secs = parse_timestamp(ts)
            if secs > 0:
                timestamps.append(secs)
        else:
            ts = s.get("timestamp", "")
            secs = parse_timestamp(ts)
            if secs > 0:
                timestamps.append(secs)

    if len(timestamps) < 2:
        return False

    span_hours = (max(timestamps) - min(timestamps)) / 3600.0
    return span_hours >= config.MIN_EVIDENCE_SPAN_HOURS


def run_temporal_check(questions, captions):
    """对每道题检查时间跨度。"""
    passed = []
    failed = []

    for q in questions:
        qid = q.get("id", "?")
        if check_temporal_span(q, captions):
            passed.append(q)
        else:
            print(f"    时间跨度不足: {qid}")
            failed.append(q)

    return passed, failed


# ================================================================
# 过滤 3: Trace Pattern 分布检查
# ================================================================

def check_trace_pattern_distribution(questions):
    """检查 trace pattern 分布是否均衡。"""
    dist = defaultdict(int)
    for q in questions:
        tp = q.get("trace_pattern", "unknown")
        dist[tp] += 1

    print(f"    Trace Pattern 分布: {dict(dist)}")

    issues = []
    # 每种 pattern 至少出现一次
    expected_patterns = {"single_scan", "cross_session",
                         "pattern_abstraction", "multi_entity"}
    missing = expected_patterns - set(dist.keys())
    if missing:
        issues.append(f"缺少 trace pattern: {missing}")

    return issues


# ================================================================
# 主流程
# ================================================================

def run_filter_pipeline(input_path=None, dry_run=False):
    """
    自动过滤主流程。

    参数:
        input_path: QA 文件路径（默认 qa_pairs.json）
        dry_run: 只统计不调用 API

    返回:
        dict: 过滤统计
    """
    if input_path is None:
        input_path = os.path.join(config.OUTPUT_DIR, "qa_pairs.json")
        if not os.path.exists(input_path):
            input_path = os.path.join(config.OUTPUT_DIR, "qa_pairs.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到 QA 文件: {input_path}")

    with open(input_path) as f:
        qa_data = json.load(f)

    questions = qa_data.get("questions", [])
    if not questions:
        print("  ⚠ 没有题目可过滤")
        return {"filtered": 0}

    print(f"  输入: {len(questions)} 道题")
    print(f"  来源: {input_path}")

    if dry_run:
        mcq_count = sum(1 for q in questions if q.get("type") == "multiple_choice")
        print(f"\n  [DRY-RUN] 过滤流程:")
        print(f"    1. 盲测: {mcq_count} 道 MCQ × {len(config.BLIND_TEST_MODELS)} 模型")
        print(f"       = {mcq_count * len(config.BLIND_TEST_MODELS)} 次 LLM 调用, ~$0.005")
        print(f"    2. 时间跨度检查: ≥ {config.MIN_EVIDENCE_SPAN_HOURS}h")
        print(f"    3. Trace Pattern 分布检查")
        return {"dry_run": True}

    # 加载 captions 用于时间检查
    captions = {}
    captions_path = os.path.join(config.CAPTIONS_DIR,
                                 "all_double_layer_captions.json")
    if os.path.exists(captions_path):
        with open(captions_path) as f:
            captions = json.load(f)

    # ---- 过滤 1: 盲测 ----
    print(f"\n  [1/3] 文本泄漏盲测...")
    questions, blind_leaked = run_blind_tests(questions)
    print(f"    结果: {len(questions)} 通过, {len(blind_leaked)} 泄漏")

    # ---- 过滤 2: 时间跨度 ----
    print(f"\n  [2/3] 时间跨度检查 (≥ {config.MIN_EVIDENCE_SPAN_HOURS}h)...")
    questions, temporal_failed = run_temporal_check(questions, captions)
    print(f"    结果: {len(questions)} 通过, {len(temporal_failed)} 时间跨度不足")

    # ---- 过滤 3: Trace Pattern 分布 ----
    print(f"\n  [3/3] Trace Pattern 分布检查...")
    pattern_issues = check_trace_pattern_distribution(questions)
    if pattern_issues:
        for issue in pattern_issues:
            print(f"    ⚠ {issue}")

    # ---- 保存过滤结果 ----
    filtered_output = {
        "original_count": qa_data.get("total_questions", len(questions)),
        "filtered_count": len(questions),
        "filtered_out": {
            "blind_leak": len(blind_leaked),
            "temporal_span": len(temporal_failed),
            "total": len(blind_leaked) + len(temporal_failed)
        },
        "by_type": qa_data.get("by_type", {}),
        "by_trace_pattern": defaultdict(int),
        "pattern_issues": pattern_issues,
        "questions": questions,
        "filtered_out_questions": {
            "blind_leak": [q["id"] for q in blind_leaked],
            "temporal_span": [q["id"] for q in temporal_failed]
        }
    }

    for q in questions:
        tp = q.get("trace_pattern", "unknown")
        filtered_output["by_trace_pattern"][tp] += 1

    output_path = os.path.join(config.OUTPUT_DIR, "qa_pairs_filtered.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 过滤完成: {len(questions)}/{qa_data.get('total_questions', '?')} 通过")
    print(f"    盲测泄漏: {len(blind_leaked)}")
    print(f"    时间跨度不足: {len(temporal_failed)}")
    print(f"  输出: {output_path}")

    return {
        "original": qa_data.get("total_questions", 0),
        "passed": len(questions),
        "blind_leak": len(blind_leaked),
        "temporal_fail": len(temporal_failed),
        "output": output_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 5: 自动过滤")
    parser.add_argument("--input", type=str, default=None,
                        help="QA 文件路径")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 5: 自动过滤 & 质量保证")
    print("=" * 60)
    run_filter_pipeline(input_path=args.input, dry_run=args.dry_run)
