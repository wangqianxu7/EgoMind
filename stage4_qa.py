"""
Stage 4: 三步问题生成
========================
Phase 2 改进 — 借鉴 EGOMEMREASON 的 statement→query→distractor 流程，
并加入 EgoMind 独有的 Trace Pattern 绑定 + 记忆混淆型干扰项。

三步流程：
  Step 1 — Evidence Statement 提取：从双层 caption 提取每个字段的事实陈述
  Step 2 — Query 生成：基于 statements 生成问题，绑定 Trace Pattern
  Step 3 — Distractor 生成：记忆混淆型干扰项（时间/人物/频率/近因）

成本估算（纯文本 LLM，极低）：
  每字段 3 次 LLM 调用 × 5 字段 = 15 次
  约 $0.02 总计

使用方式：
  python stage5_qa_v2.py                  # 从 profile + captions 生成 QA
  python stage5_qa_v2.py --dry-run        # 只看不跑
"""

import json
import os
import time
from openai import OpenAI
import config

# 输出路径
QA_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "qa_pairs.json")


# ================================================================
# Step 1: Evidence Statement 提取
# ================================================================

from prompts import (STAGE4_STEP1_SYSTEM, STAGE4_STEP1_USER,
                     STAGE4_STEP2_SYSTEM, STAGE4_STEP2_USER)


def extract_statements(client, field_info, captions):
    """Step 1: 从 caption 中提取 evidence statements。"""
    dim, fld = field_info["dimension"], field_info["field"]
    dim_config = config.PREFERENCE_DIMENSIONS.get(dim, {})
    desc = dim_config.get("description", f"{dim}.{fld}")

    # 构建 caption 摘要文本
    caption_lines = []
    for clip_id, cap in sorted(captions.items()):
        l1 = cap.get("layer1", {})
        l2 = cap.get("layer2", {})
        dim_signals = l2.get(dim, {})

        summary = (
            f"[{clip_id} @ {cap['timestamp']}] "
            f"scene: {l1.get('scene_summary', '?')[:100]}; "
            f"signals({dim}): {json.dumps(dim_signals, ensure_ascii=False)}"
        )
        caption_lines.append(summary)

    # 限制长度避免超 context
    captions_text = "\n".join(caption_lines[:200])

    user_msg = STAGE4_STEP1_USER.format(
        dimension=dim, field=fld, description=desc,
        captions_text=captions_text,
        max_statements=config.MAX_STATEMENTS_PER_FIELD
    )

    try:
        resp = client.chat.completions.create(
            model=config.TEXT_LLM_MODEL,
            messages=[
                {"role": "system", "content": STAGE4_STEP1_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=config.TEXT_LLM_MAX_TOKENS,
            temperature=config.TEXT_LLM_TEMPERATURE_EXTRACT,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content.strip())
        return result.get("statements", [])
    except Exception as e:
        print(f"    ✗ Step 1 失败: {e}")
        return []


# ================================================================
# Step 2: Query 生成（绑定 Trace Pattern）
# ================================================================

TRACE_PATTERNS = {
    "single_scan": {
        "name": "Single-scan",
        "description": "Direct preference inference from a few closely-related clips",
        "example": "Does Jake prefer KFC or hotpot based on his lunch choices?"
    },
    "cross_session": {
        "name": "Cross-session",
        "description": "Comparison across different time periods requiring SCAN of two ranges",
        "example": "How does Jake's food choice differ between lunch and dinner?"
    },
    "pattern_abstraction": {
        "name": "Pattern-abstraction",
        "description": "Abstract recurring patterns needing multiple ZOOM confirmations",
        "example": "What brand appears most consistently across all meals this week?"
    },
    "multi_entity": {
        "name": "Multi-entity",
        "description": "Track object/people flow across multiple clips",
        "example": "After Jake passes his phone, who typically receives it?"
    }
}

MEMORY_CONFUSION_TYPES = [
    {
        "type": "temporal_confusion",
        "description": "Correct action/object but wrong time (e.g., dinner item at lunch)"
    },
    {
        "type": "person_confusion",
        "description": "Correct event but wrong person (e.g., gave item to Alice not Tasha)"
    },
    {
        "type": "frequency_illusion",
        "description": "Single occurrence mistaken as habit (e.g., ate KFC once → always eats KFC)"
    },
    {
        "type": "recency_bias",
        "description": "Most recent behavior mistaken as typical (e.g., last seen = default)"
    }
]

# STAGE4_STEP2_SYSTEM / STAGE4_STEP2_USER imported from prompts above


def generate_queries(client, field_info, statements, captions):
    """Step 2: 基于 statements 生成问题。"""
    dim, fld = field_info["dimension"], field_info["field"]

    # 格式化 statements
    stmts_text = "\n".join(
        f"  [{s.get('clip_id', '?')} @ {s.get('timestamp', '?')}] "
        f"[{s.get('type', '?')}] {s.get('text', '')}"
        for s in statements
    ) if statements else "(No statements extracted — generate from captions context)"

    # Captions 摘要
    cap_lines = []
    for clip_id, cap in list(sorted(captions.items()))[:3]:
        l1 = cap.get("layer1", {})
        cap_lines.append(
            f"[{clip_id}] {l1.get('scene_summary', cap.get('srt_reference', ''))[:200]}"
        )
    captions_summary = "\n".join(cap_lines)

    trace_options = ", ".join(TRACE_PATTERNS.keys())
    question_types = (
        "at least 1 multiple_choice + 1 open_ended, "
        "with distinct trace_pattern assignments"
    )

    user_msg = STAGE4_STEP2_USER.format(
        dimension=dim, field=fld,
        value=field_info.get("value", "unknown"),
        confidence=field_info.get("confidence", 0.5),
        statements_text=stmts_text,
        captions_summary=captions_summary,
        num_questions=config.QUESTIONS_PER_FIELD,
        question_types=question_types,
        trace_options=trace_options
    )

    try:
        resp = client.chat.completions.create(
            model=config.TEXT_LLM_MODEL,
            messages=[
                {"role": "system", "content": STAGE4_STEP2_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=config.TEXT_LLM_MAX_TOKENS,
            temperature=config.TEXT_LLM_TEMPERATURE_ORACLE,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content.strip())
        return result.get("questions", [])
    except Exception as e:
        print(f"    ✗ Step 2 失败: {e}")
        return []


def _clean_option_labels(q: dict):
    """去掉选项文本中的 (type_label) 标记, distractor_types 元数据保留。"""
    import re
    if "options" in q:
        q["options"] = [re.sub(r"\s*\([a-z_]+\)\s*", "", o).strip()
                        for o in q["options"]]


# ================================================================
# 主流程
# ================================================================

def generate_questions(dry_run=False):
    """
    三步问题生成主流程。

    流程:
      1. 加载 profile + 双层 captions
      2. 对每个字段：提取 statements → 生成 queries → 标注元数据
      3. 保存 qa_pairs.json
    """
    # 加载 profile
    profile_path = os.path.join(config.PROFILE_DIR, "M_star.json")
    with open(profile_path) as f:
        profile = json.load(f)
    verified = profile.get("profile", {})

    # 加载双层 captions（fallback 到 SRT captions）
    captions_path = os.path.join(config.CAPTIONS_DIR,
                                    "all_double_layer_captions.json")
    captions_path_v1 = os.path.join(config.CAPTIONS_DIR, "all_captions.json")

    if os.path.exists(captions_path):
        with open(captions_path) as f:
            captions = json.load(f)
        print(f"  使用双层 captions: {len(captions)} clips")
    elif os.path.exists(captions_path_v1):
        with open(captions_path_v1) as f:
            captions = json.load(f)
        print(f"  ⚠ 双层 captions 不存在，fallback 到 SRT captions: {len(captions)} clips")
    else:
        raise FileNotFoundError("未找到任何 captions 文件，请先运行 Stage 2")

    # 收集已填充字段
    populated_fields = []
    for dim_name, fields in verified.items():
        for field_name, field_data in fields.items():
            if field_data and field_data.get("value") is not None:
                populated_fields.append({
                    "dimension": dim_name,
                    "field": field_name,
                    "value": field_data["value"],
                    "confidence": field_data.get("confidence", 0.5),
                    "evidence": field_data.get("evidence", [])
                })

    if not populated_fields:
        print("  ⚠ 没有已填充的 profile 字段，无法生成问题")
        return {"num_questions": 0, "qa_path": QA_OUTPUT_PATH, "by_type": {}}

    print(f"  {len(populated_fields)} 个已填充字段")

    if dry_run:
        est_calls = len(populated_fields) * 2  # Step 1 + Step 2 per field
        print(f"  [DRY-RUN] 预估 {est_calls} 次 LLM 调用，~$0.01")
        return {"dry_run": True, "num_fields": len(populated_fields)}

    client = OpenAI(
        api_key=config.TEXT_LLM_API_KEY,
        base_url=config.TEXT_LLM_BASE_URL if config.TEXT_LLM_BASE_URL else None
    )

    all_questions = []
    stats = {"direct_preference": 0, "habit_inference": 0, "cross_session": 0}
    trace_pattern_stats = {k: 0 for k in TRACE_PATTERNS}
    q_counter = [0]

    for idx, field_info in enumerate(populated_fields):
        dim, fld = field_info["dimension"], field_info["field"]
        print(f"\n  [{idx+1}/{len(populated_fields)}] {dim}.{fld}")

        # Step 1: 提取 statements
        print(f"    Step 1: 提取 evidence statements...", end=" ")
        statements = extract_statements(client, field_info, captions)
        print(f"✓ ({len(statements)} statements)")

        # Step 2: 生成问题（含 distractor）
        print(f"    Step 2: 生成问题...", end=" ")
        questions = generate_queries(client, field_info, statements, captions)

        if not questions:
            print("✗ (无问题)")
            continue

        # 标注元数据
        for q in questions:
            q_counter[0] += 1
            q["id"] = f"q_{q_counter[0]:03d}"
            q["source_dimension"] = dim
            q["source_field"] = fld
            q["source_value"] = field_info["value"]
            q["source_confidence"] = field_info["confidence"]
            q["evidence_statements"] = statements

            rtype = q.get("reasoning_type", "direct_preference")
            if rtype in stats:
                stats[rtype] += 1

            tp = q.get("trace_pattern", "")
            if tp in trace_pattern_stats:
                trace_pattern_stats[tp] += 1

        for q in questions:
            _clean_option_labels(q)
        all_questions.extend(questions)
        print(f"✓ ({len(questions)} questions)")

    # 保存
    qa_output = {
        "total_questions": len(all_questions),
        "by_type": stats,
        "by_trace_pattern": trace_pattern_stats,
        "version": "1.0",
        "questions": all_questions
    }
    os.makedirs(os.path.dirname(QA_OUTPUT_PATH), exist_ok=True)
    with open(QA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 总计: {len(all_questions)} 个问题")
    print(f"  类型分布: {stats}")
    print(f"  Trace Pattern 分布: {trace_pattern_stats}")
    print(f"  输出: {QA_OUTPUT_PATH}")

    return {
        "num_questions": len(all_questions),
        "qa_path": QA_OUTPUT_PATH,
        "by_type": stats,
        "by_trace_pattern": trace_pattern_stats
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: 三步问题生成")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 4: 三步问题生成")
    print("=" * 60)
    generate_questions(dry_run=args.dry_run)
