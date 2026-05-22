"""
Stage 7: 验证、Sparsity 评分、训练样本输出
============================================
对每条 inquiry trace 执行三步验证，
对通过的 trace 计算 sparsity score σ(q)，
按 σ 中位数划分 EgoPref-SFT 和 EgoPref-RL，
最终输出可用于训练的 JSON 样本。

对应论文: EgoPref §2.5-2.7
  验证 1: SCAN/ZOOM 步骤的 clip 相关性  |C^(t) ∩ C*_q| ≥ 1
  验证 2: REVISE 操作的正确方向 δ(M_after, M*) < δ(M_before, M*)
  验证 3: 最终答案正确性
  Sparsity: σ(q) = -log(r_q/N_P) · (s_q/r_q)
  划分: σ ≤ median → SFT,  σ > median → RL
"""

import json
import math
import os
from typing import Optional
import config


def load_traces() -> dict:
    path = os.path.join(config.TRACES_DIR, "all_traces.json")
    if not os.path.exists(path):
        raise FileNotFoundError("all_traces.json 不存在, 请先运行 Stage 6")
    with open(path) as f:
        return json.load(f)


def load_profile() -> dict:
    with open(os.path.join(config.PROFILE_DIR, "M_star.json")) as f:
        return json.load(f)


def load_captions() -> dict:
    captions_path = os.path.join(config.CAPTIONS_DIR,
                           "all_double_layer_captions.json")
    v1_path = os.path.join(config.CAPTIONS_DIR, "all_captions.json")
    path = captions_path if os.path.exists(captions_path) else v1_path
    with open(path) as f:
        captions = json.load(f)
    for cap in captions.values():
        if "description" not in cap:
            cap["description"] = _fmt_caption(cap)
    return captions


def _fmt_caption(cap: dict) -> str:
    l1 = cap.get("layer1", {})
    parts = []
    if l1.get("scene_summary"):
        parts.append(l1["scene_summary"])
    if l1.get("objects"):
        obj_strs = []
        for obj in l1["objects"]:
            name = obj.get("name", "?")
            states = obj.get("states", [])
            obj_strs.append(f"{name}[{', '.join(states)}]" if states else name)
        parts.append("Objects: " + "; ".join(obj_strs))
    if l1.get("actions"):
        parts.append("Actions: " + " → ".join(l1["actions"]))
    if l1.get("people"):
        parts.append("People: " + ", ".join(l1["people"]))
    srt = cap.get("srt_reference", "")
    if srt:
        parts.append(f"SRT: {srt[:200]}")
    return " | ".join(parts)


def load_clip_metadata() -> list:
    # EgoLife 模式：从数据目录加载
    for identity in config.EGOLIFE_IDENTITIES:
        for day in config.EGOLIFE_DAYS:
            path = os.path.join(config.EGOLIFE_DATA_DIR, identity, day, "clips_metadata.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
    # fallback: YouTube 旧路径
    with open(os.path.join(config.OUTPUT_DIR, "clip_metadata.json")) as f:
        return json.load(f)


def extract_clip_id(evidence) -> str:
    """从 evidence 条目提取 clip_id。

    支持两种格式：
      - dict: {"clip_id": "DAY1_A1_JAKE_13370000", ...}
      - str:  "DAY1_A1_JAKE_13370000: 好那就KFC吧怎么样"
    """
    if isinstance(evidence, dict):
        return evidence.get("clip_id", "")
    # 字符串格式: "CLIP_ID: excerpt..." → 取冒号前的 clip_id
    return str(evidence).split(":")[0].strip()


# ================================================================
# Step 1: 验证 examined clips 的相关性
# 论文 Eq.3: |C^(t) ∩ C*_q| ≥ 1
# ================================================================
def verify_examine_relevance(examined_clips: list,
                             ground_truth_clips: set) -> bool:
    """
    检查每个 SCAN/ZOOM 步骤检索到的 clip 是否至少有一个相关。

    论文: "A SCAN or ZOOM step is accepted iff |C^(t) ∩ C*_q| ≥ 1"

    对于本实验（单视频 10 clips 级别），ground-truth relevant set
    由 profile 提取时引用的 evidence clip_ids 构成。
    """
    if not examined_clips or not ground_truth_clips:
        return False
    overlap = set(examined_clips) & ground_truth_clips
    return len(overlap) >= 1


# ================================================================
# Step 2: 验证 REVISE 操作的正确方向
# 论文 Eq.4: δ(M_before, M*) - δ(M_after, M*) > 0
# ================================================================
def compute_field_disagreement(memory_state: dict, ground_truth: dict) -> float:
    """
    计算 partial memory 与 ground-truth profile 在 memory 已覆盖字段上的差异。

    只比较 memory_state 中实际存在的字段（其他维度/字段不需要填充）。
    论文: δ(·, M*_P) is the field-level disagreement measure
          对 categorical: exact match; 对 continuous: tolerance-based
    """
    if not memory_state or not ground_truth:
        return 1.0

    total_fields = 0
    disagreements = 0

    for dim_name, mem_dim in memory_state.items():
        if not isinstance(mem_dim, dict):
            continue
        gt_dim = ground_truth.get(dim_name, {})
        for field_name, mem_data in mem_dim.items():
            if not isinstance(mem_data, dict):
                continue
            mem_val = mem_data.get("value")
            if mem_val is None:
                continue
            gt_data = gt_dim.get(field_name, {})
            gt_val = gt_data.get("value") if gt_data else None
            if gt_val is None:
                continue

            total_fields += 1

            # 标准化比较
            def norm(v):
                if isinstance(v, list):
                    return sorted([str(x).lower().strip() for x in v])
                return str(v).lower().strip() if v is not None else ""

            if norm(mem_val) != norm(gt_val):
                disagreements += 1

    return disagreements / max(total_fields, 1)


def verify_revise_direction(memory_before: dict, memory_after: dict,
                            ground_truth: dict) -> bool:
    """
    论文 Eq.4: δ(M^(t-1)_P, M*_P) - δ(M^(t)_P, M*_P) ≥ 0
    即每次 REVISE 不能让 memory 变差（更远离 ground truth）。
    """
    delta_before = compute_field_disagreement(memory_before, ground_truth)
    delta_after = compute_field_disagreement(memory_after, ground_truth)
    return delta_after <= delta_before


# ================================================================
# Step 3: 验证最终答案
# ================================================================
def verify_final_answer(trace_rounds: list, ground_truth: str) -> bool:
    """
    检查 COMMIT 步骤的 answer 是否匹配 ground truth。

    论文: "COMMIT 时的 answer 必须等于 ground-truth answer"
    容错: 去掉首尾空格、统一大小写、多选题选项字母匹配
    """
    if not trace_rounds:
        return False

    last_round = trace_rounds[-1]
    action = last_round.get("action", {})
    if isinstance(action, dict):
        action_type = action.get("type", "")
        if action_type not in ("A_commit", "COMMIT", "commit"):
            return False
        predicted = action.get("params", {}).get("answer", "")
    elif isinstance(action, str):
        predicted = action
    else:
        return False

    # 规范化比较
    pred_clean = str(predicted).strip().upper()
    gt_clean = str(ground_truth).strip().upper()
    return pred_clean == gt_clean or gt_clean in pred_clean or pred_clean in gt_clean


# ================================================================
# Sparsity Score 计算
# 论文 Eq.5: σ(q) = -log(r_q / N_P) · (s_q / r_q)
# ================================================================
def compute_sparsity(trace_data: dict, clip_metadata: list,
                     captions: dict, profile: dict) -> float:
    """
    计算问题的 sparsity score。

    论文公式: σ(q) = -log(r_q / N_P) · (s_q / r_q)

    r_q: 与问题相关的 clip 数量（从 evidence clips 推导）
    s_q: 相关 clip 跨多少个不同 session（此处简化为不同 clip）
    N_P: 总 clip 数

    说明: 对于单视频实验，没有真正的 session 概念。
          用 clip 本身作为 session 的代理。
          在实际 EgoPref 中，session = 一天的录像。
    """
    N_P = len(clip_metadata)

    # ---- 确定相关 clip 集合 C*_q ----
    # 从 profile 中找出与该问题 source_field 匹配的 evidence clips
    source_field = trace_data.get("source_field", "")
    relevant_clips = set()

    for dim_name, fields in profile.get("profile", {}).items():
        for field_name, field_data in fields.items():
            if field_name == source_field and field_data:
                for ev in field_data.get("evidence", []):
                    cid = extract_clip_id(ev)
                    if cid:
                        relevant_clips.add(cid)

    r_q = max(len(relevant_clips), 1)  # 至少为 1, 避免 log(0)

    # ---- 计算 distinct sessions ----
    # 单视频场景: 每个 clip 就是一个 "session"
    # 实际使用中按天的 session 划分才有意义
    s_q = len(relevant_clips)

    # ---- 计算 sparsity ----
    rarity = -math.log(r_q / N_P) if r_q > 0 else 0
    dispersion = s_q / r_q if r_q > 0 else 0
    sigma = rarity * dispersion

    return sigma


def verify_and_score() -> dict:
    """
    主函数: 验证所有 traces + 计算 sparsity + 划分 + 输出训练样本。

    流程:
      1. 加载所有中间产物
      2. 逐条验证 trace（三步检查）
      3. 对通过的 trace 计算 σ(q)
      4. 按 σ 中位数划分为 SFT / RL
      5. 输出 JSON 训练样本到 training/
    """
    traces = load_traces()
    profile = load_profile()
    captions = load_captions()
    clip_metadata = load_clip_metadata()

    # ---- 收集 ground truth evidence clips ----
    gt_clips = set()
    for dim_data in profile.get("profile", {}).values():
        for field_data in dim_data.values():
            if field_data and field_data.get("evidence"):
                for ev in field_data["evidence"]:
                    cid = extract_clip_id(ev)
                    if cid:
                        gt_clips.add(cid)

    verified_traces = []
    rejected_traces = []
    stats = {"examined": 0, "relevance_fail": 0, "revise_fail": 0,
             "answer_fail": 0, "passed": 0}

    print(f"  总 traces: {len(traces)}")
    print(f"  Ground-truth evidence clips: {gt_clips}")

    # ================================================================
    # 逐条验证
    # ================================================================
    for qid, trace_data in traces.items():
        stats["examined"] += 1
        rounds = trace_data.get("rounds", [])

        # ---- 检查 1: 每个 examine step 的 clip 相关性 ----
        relevance_ok = True
        for rnd in rounds:
            action = rnd.get("action", {})
            action_type = action.get("type", "") if isinstance(action, dict) else str(action)
            if action_type in ("A_scan", "A_zoom", "SCAN", "ZOOM"):
                examined = rnd.get("examined_clips", [])
                if not verify_examine_relevance(examined, gt_clips):
                    relevance_ok = False
                    break

        if not relevance_ok:
            stats["relevance_fail"] += 1
            rejected_traces.append({"qid": qid, "reason": "relevance_fail"})
            continue

        # ---- 检查 2: REVISE 方向 ----
        # 首轮 (i=0) 的 mem_before 总是空的，跳过方向检查
        # 只验证后续轮的 REVISE 是否让 memory 更接近 ground truth
        revise_ok = True
        for i, rnd in enumerate(rounds):
            if i == 0:
                continue
            mem_before = rounds[i-1].get("memory_state_after", {})
            mem_after = rnd.get("memory_state_after", {})
            mem_ops = rnd.get("memory_operations", [])
            has_revise = any(
                op.get("op") in ("R_rev", "REVISE") for op in mem_ops
            )
            if has_revise and mem_after:
                if not verify_revise_direction(
                    mem_before, mem_after, profile.get("profile", {})
                ):
                    revise_ok = False
                    break

        if not revise_ok:
            stats["revise_fail"] += 1
            rejected_traces.append({"qid": qid, "reason": "revise_fail"})
            continue

        # ---- 检查 3: 最终答案 ----
        if not verify_final_answer(rounds, trace_data.get("ground_truth", "")):
            stats["answer_fail"] += 1
            rejected_traces.append({"qid": qid, "reason": "answer_fail"})
            continue

        # ---- 通过！计算 sparsity ----
        sigma = compute_sparsity(trace_data, clip_metadata, captions, profile)
        trace_data["sparsity_score"] = round(sigma, 4)
        trace_data["verified"] = True
        verified_traces.append(trace_data)
        stats["passed"] += 1

    # ================================================================
    # Sparsity 中位数划分 → SFT / RL
    # 论文: EgoPref-SFT (σ ≤ median), EgoPref-RL (σ > median)
    # ================================================================
    if verified_traces:
        sigmas = [t["sparsity_score"] for t in verified_traces]
        median_sigma = sorted(sigmas)[len(sigmas) // 2]

        sft_traces = [t for t in verified_traces if t["sparsity_score"] <= median_sigma]
        rl_traces = [t for t in verified_traces if t["sparsity_score"] > median_sigma]

        print(f"\n  验证结果:")
        print(f"    总数: {stats['examined']}, 通过: {stats['passed']}")
        print(f"    相关性失败: {stats['relevance_fail']}")
        print(f"    REVISE 方向失败: {stats['revise_fail']}")
        print(f"    答案错误: {stats['answer_fail']}")
        print(f"    Pass rate: {stats['passed']/max(stats['examined'],1)*100:.1f}%")
        print(f"  Sparsity median: {median_sigma:.4f}")
        print(f"  EgoPref-SFT: {len(sft_traces)} traces")
        print(f"  EgoPref-RL: {len(rl_traces)} traces")

        # ---- 输出最终训练样本 ----
        for split_name, split_data in [("SFT", sft_traces), ("RL", rl_traces)]:
            split_dir = os.path.join(config.TRAINING_DIR, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for trace in split_data:
                qid = trace.get("question_id", "unknown")
                # 最终训练样本格式 —— 对应论文 Table 3
                instance = {
                    "instance_id": f"{qid}",
                    "individual_id": "subject_001",
                    "question": trace["question"],
                    "question_type": trace.get("question_type", ""),
                    "ground_truth_answer": trace["ground_truth"],
                    "initial_memory_state": {},
                    "expected_trace": trace["rounds"],
                    "sparsity_score": trace["sparsity_score"],
                    "split": split_name,
                }
                fpath = os.path.join(split_dir, f"{qid}.json")
                with open(fpath, "w") as f:
                    json.dump(instance, f, indent=2, ensure_ascii=False)

            # ---- 写 split 级别的汇总 JSON ----
            summary_path = os.path.join(config.TRAINING_DIR, f"egopref_{split_name.lower()}.json")
            with open(summary_path, "w") as f:
                json.dump([json.load(open(os.path.join(split_dir, fn)))
                           for fn in sorted(os.listdir(split_dir))],
                          f, indent=2, ensure_ascii=False)

    # ---- 保存验证报告 ----
    report = {
        "verification_stats": stats,
        "num_verified": len(verified_traces),
        "num_rejected": len(rejected_traces),
        "rejected_details": rejected_traces,
        "sparsity_median": median_sigma if verified_traces else None,
        "sft_count": len(sft_traces) if verified_traces else 0,
        "rl_count": len(rl_traces) if verified_traces else 0
    }
    report_path = os.path.join(config.OUTPUT_DIR, "verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "verified": stats["passed"],
        "total": stats["examined"],
        "pass_rate": f"{stats['passed']/max(stats['examined'],1)*100:.1f}%",
        "sft_count": len(sft_traces) if verified_traces else 0,
        "rl_count": len(rl_traces) if verified_traces else 0,
        "report_path": report_path
    }


if __name__ == "__main__":
    verify_and_score()
