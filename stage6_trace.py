"""
Stage 6: Oracle 合成 Multi-step Inquiry Traces
================================================
对每个问题，LLM oracle 模拟理想推理者，生成多步 inquiry trace。
每步包含 <observe> → <remember> → <infer> → <action>。

对应论文: EgoPref §2.4 Trace Synthesis
  - Oracle 知道 ground truth profile（用于验证）
  - 但必须模拟"不知道答案的 agent 应该怎么推理"
  - 3 种 inquiry action: A_scan / A_zoom / A_commit
  - 3 种 memory operation: R_rec / R_rev / R_refl
  - 每步 ≤T_MAX=3 步

Trace 结构 (论文 Fig.2):
  Round t: {
    <observe>:  分析本轮检索到的 clip
    <remember>: 更新 memory (R_rev 操作 + evidence 引用)
    <infer>:    推理目前已知了什么，下一步方向
    <action>:   下一动作 (A_scan / A_zoom / A_commit)
  }
"""

import json
import os
import time
from openai import OpenAI
import config
from prompts import STAGE6_SYSTEM, STAGE6_USER


def load_qa_pairs() -> list:
    # 优先级: filtered → qa_pairs
    for fname in ["qa_pairs_filtered.json", "qa_pairs_v2.json", "qa_pairs.json"]:
        path = os.path.join(config.OUTPUT_DIR, fname)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError("未找到 QA pairs，请先运行 Stage 4 + Stage 5")
    with open(path) as f:
        data = json.load(f)
    return data.get("questions", data.get("filtered_questions", []))


def load_captions() -> dict:
    # 优先
    captions_path = os.path.join(config.CAPTIONS_DIR,
                           "all_double_layer_captions.json")
    v1_path = os.path.join(config.CAPTIONS_DIR, "all_captions.json")
    path = captions_path if os.path.exists(captions_path) else v1_path
    with open(path) as f:
        captions = json.load(f)
    # 为每个 caption 注入 description 文本（兼容结构化格式）
    for cap in captions.values():
        if "description" not in cap:
            cap["description"] = _fmt_caption(cap)
    return captions


def _fmt_caption(cap: dict) -> str:
    """结构化 caption → 文本行。"""
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


def build_trace_prompt(q: dict, captions: dict, profile: dict) -> tuple:
    """
    构建 trace 合成的 prompt。

    论文对应: Table 2 - Structured prompt for inquiry trace synthesis
    关键约束:
      - Oracle 知道答案但必须"假装不知道"
      - 每一步都是 evidence discovery
      - 只有当证据明确支持才 A_commit
      - confidence 要反映证据强度
    """
    # ---- 提取该问题对应字段的 evidence clip IDs ----
    source_field = q.get("source_field", "")
    evidence_clip_ids = set()
    for dim_name, fields in profile.get("profile", {}).items():
        for field_name, field_data in fields.items():
            if field_name == source_field and field_data:
                for ev in field_data.get("evidence", []):
                    if isinstance(ev, dict):
                        cid = ev.get("clip_id", "")
                    else:
                        cid = str(ev).split(":")[0].strip()
                    if cid:
                        evidence_clip_ids.add(cid)

    # ---- 整理 evidence clips 的详细信息 ----
    evidence_info = ""
    if evidence_clip_ids:
        ev_lines = []
        for cid in sorted(evidence_clip_ids):
            if cid in captions:
                desc = captions[cid]["description"][:150].replace("\n", " ")
                ev_lines.append(f"  [{cid}] {desc}")
        if ev_lines:
            evidence_info = (
                f"\nEVIDENCE CLIPS (these contain the ground-truth answer — "
                f"the agent should DISCOVER them through search):\n"
                + "\n".join(ev_lines) + "\n"
            )

    # ---- 整理 clip 信息（ID + 描述摘要） ----
    clip_lines = []
    for clip_id, cap in sorted(captions.items()):
        desc_short = cap["description"][:120].replace("\n", " ")
        clip_lines.append(f"  [{clip_id} @ {cap['timestamp']}] {desc_short}...")
    clip_info = "\n".join(clip_lines)

    options = q.get('options', [])
    options_line = f"OPTIONS: {options}\n" if options else ""

    system_msg = STAGE6_SYSTEM.format(t_max=config.T_MAX)
    user_msg = STAGE6_USER.format(
        question=q['question'],
        qtype=q.get('type', 'multiple_choice'),
        options_line=options_line,
        ground_truth=q.get('ground_truth'),
        evidence_info=evidence_info,
        n_clips=len(captions),
        clip_info=clip_info,
        t_max=config.T_MAX,
        scan_k=config.SCAN_K,
    )
    return system_msg, user_msg


def synthesize_traces() -> dict:
    """
    主函数：为每个 QA pair 合成 inquiry trace。

    流程:
      1. 加载 QA pairs + captions + profile
      2. 对每个问题调用 GPT-4o-mini Oracle
      3. Oracle 模拟逐步推理 → 产出 1-3 步 trace
      4. 存入 traces/ 目录

    返回:
        dict: {"num_traces": int, "traces_dir": str}
    """
    questions = load_qa_pairs()
    captions = load_captions()

    # 加载 profile 摘要（给 Oracle 做 ground truth 参考）
    profile_path = os.path.join(config.PROFILE_DIR, "M_star.json")
    with open(profile_path) as f:
        profile = json.load(f)

    print(f"  准备为 {len(questions)} 个问题生成 inquiry traces")

    client = OpenAI(
        api_key=config.TEXT_LLM_API_KEY,
        base_url=config.TEXT_LLM_BASE_URL if config.TEXT_LLM_BASE_URL else None
    )

    traces = {}
    step_counts = []

    # ---- 只处理前 MAX_QUESTIONS 个问题来管理成本 ----
    MAX_QUESTIONS = min(len(questions), 15)
    questions = questions[:MAX_QUESTIONS]
    print(f"  实际处理 {MAX_QUESTIONS} 个（受成本限制）")

    for idx, q in enumerate(questions):
        qid = q.get("id", f"q_{idx:04d}")
        print(f"  [{idx+1}/{MAX_QUESTIONS}] 合成 {qid} 的 trace...", end=" ")

        system_msg, user_msg = build_trace_prompt(q, captions, profile)

        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=config.TEXT_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=config.TEXT_LLM_MAX_TOKENS,
                    temperature=config.TEXT_LLM_TEMPERATURE_ORACLE,
                    response_format={"type": "json_object"}
                )
                result = json.loads(resp.choices[0].message.content.strip())
                break
            except (json.JSONDecodeError, Exception) as e:
                if attempt == config.LLM_MAX_RETRIES - 1:
                    print(f"✗ ({e})")
                    result = None
                time.sleep(2 ** attempt)

        if result is None:
            continue

        # ---- 解析 trace ----
        trace_rounds = result.get("trace", [])
        n_steps = len(trace_rounds)
        step_counts.append(n_steps)

        # ---- 丰富 trace 元数据 ----
        trace_data = {
            "question_id": qid,
            "question": q["question"],
            "question_type": q.get("type"),
            "source_dimension": q.get("source_dimension"),
            "source_field": q.get("source_field"),
            "ground_truth": q.get("ground_truth"),
            "num_rounds": n_steps,
            "rounds": trace_rounds,
            "model": config.LLM_MODEL
        }
        traces[qid] = trace_data

        # ---- 存为独立文件 ----
        trace_path = os.path.join(config.TRACES_DIR, f"{qid}.json")
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        print(f"✓ ({n_steps} steps)")

    # ---- 写入总 traces 文件 ----
    traces_json_path = os.path.join(config.TRACES_DIR, "all_traces.json")
    with open(traces_json_path, "w") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)

    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0

    return {
        "num_traces": len(traces),
        "traces_dir": config.TRACES_DIR,
        "avg_steps": round(avg_steps, 1),
        "traces_json": traces_json_path
    }


if __name__ == "__main__":
    synthesize_traces()
