"""
Stage 3: LLM Oracle 提取 Ground-truth Preference Profile M*_P
==============================================================
从全部 clip captions 中提取 5 维偏好画像。
采用双重验证 (dual verification) 抑制 LLM 幻觉。

对应论文: EgoPref §2.2 Ground-truth profiles
  - 每个字段必须给出 value + confidence + ≥2 evidence clips
  - 双验证：2 次独立提取，只有一致的字段才采纳
  - 10% 人工抽检 → 87.3% agreement

双重验证流程（论文对应）:
  Pass 1: Oracle 看全部 captions → 填充 schema
  Pass 2: Oracle 只看 Pass 1 中引用的 evidence clips
           → 独立赋值（看不到 Pass 1 填的值）
  一致性检查: Pass 1 value == Pass 2 value → 采纳
"""

import json
import os
import time
from openai import OpenAI
import config
from prompts import STAGE3_SYSTEM, STAGE3_PASS1_USER, STAGE3_PASS2_USER


def load_captions_text() -> tuple[str, dict]:
    """
    加载全部双层 captions 并拼接为结构化时间线文本。

    双层 caption 格式:
      layer1: scene_summary, objects, actions, spatial, people
      layer2: preference signals (mobility/temporal/spatial/social/consumption)
      srt_reference: 原 SRT ground truth
    """
    # 优先双层 captions，fallback 到 SRT captions
    captions_path = os.path.join(config.CAPTIONS_DIR,
                           "all_double_layer_captions.json")
    v1_path = os.path.join(config.CAPTIONS_DIR, "all_captions.json")

    if not os.path.exists(captions_path):
        if os.path.exists(v1_path):
            captions_path = v1_path
        else:
            raise FileNotFoundError(
                f"未找到 captions 文件，请先运行 Stage 2\n"
                f"  查找路径: {captions_path}\n"
                f"  查找路径: {v1_path}"
            )

    with open(captions_path) as f:
        captions = json.load(f)

    # 按 clip_id 排序保证时间顺序
    sorted_clips = sorted(captions.items(), key=lambda x: x[0])

    lines = []
    for clip_id, cap in sorted_clips:
        desc = _format_caption(cap)
        # 为下游统一注入 description 字段
        cap["description"] = desc
        lines.append(f"[{clip_id}] timestamp={cap.get('timestamp', '?')}\n  {desc}")

    return "\n\n".join(lines), captions


def _format_caption(cap: dict) -> str:
    """将结构化 caption 序列化为文本行。"""
    # V1 兼容：如果已有 description 字段，直接返回
    if "description" in cap:
        return cap["description"]

    parts = []

    # Layer 1
    l1 = cap.get("layer1", {})
    if l1.get("scene_summary"):
        parts.append(l1["scene_summary"])
    if l1.get("location") or l1.get("spatial"):
        spatial = l1.get("spatial", {})
        loc_parts = []
        if spatial.get("location"):
            loc_parts.append(spatial["location"])
        if spatial.get("indoor_outdoor"):
            loc_parts.append(f"({spatial['indoor_outdoor']})")
        if loc_parts:
            parts.append("Location: " + ", ".join(loc_parts))
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

    # SRT reference
    srt = cap.get("srt_reference", "")
    if srt:
        parts.append(f"SRT: {srt[:300]}")

    return " | ".join(parts)


def call_oracle(system_msg: str, user_msg: str,
                temperature: float = None) -> str:
    """
    调用 Text LLM Oracle 的统一封装。
    使用 config.TEXT_LLM_* 配置（DeepSeek / Kimi / GPT-4o）。

    参数:
        system_msg: system prompt
        user_msg: user prompt
        temperature: 调用温度（默认用 config.TEXT_LLM_TEMPERATURE_EXTRACT）

    返回:
        str: LLM 原始响应文本
    """
    if temperature is None:
        temperature = config.TEXT_LLM_TEMPERATURE_EXTRACT

    client = OpenAI(
        api_key=config.TEXT_LLM_API_KEY,
        base_url=config.TEXT_LLM_BASE_URL if config.TEXT_LLM_BASE_URL else None
    )

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=config.TEXT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=config.TEXT_LLM_MAX_TOKENS,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == config.LLM_MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def parse_profile_json(raw: str) -> dict:
    """解析 Oracle 返回的 JSON 字符串为 Python dict，容错处理。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块 (GPT 有时会在外层包 markdown ```json )
        if "```" in raw:
            blocks = raw.split("```")
            for block in blocks:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"无法解析 Oracle 输出的 JSON: {raw[:200]}...")


def verify_fields_consistency(pass1: dict, pass2: dict) -> dict:
    """
    双验证核心：检查两次独立提取的字段值是否一致。

    论文对应: "只有两次一致的字段才被采纳"

    比较逻辑:
      - 对每个 dimension → 每个 field
      - 比较 value（字符串或列表 → 集合比较）
      - 一致的字段：保留 + 取两次中更高的 confidence
      - 不一致的字段：丢弃（留空）
    """
    verified = {}
    stats = {"total_fields": 0, "consistent": 0, "rejected": 0}

    for dim_name, dim_config in config.PREFERENCE_DIMENSIONS.items():
        verified[dim_name] = {}
        p1_dim = pass1.get(dim_name, {})
        p2_dim = pass2.get(dim_name, {})

        for field in dim_config["fields"]:
            stats["total_fields"] += 1
            v1 = p1_dim.get(field, {}).get("value")
            v2 = p2_dim.get(field, {}).get("value")

            # ---- 值一致性比较 ----
            # 将值规范化为可比较形式
            if v1 is None or v2 is None:
                stats["rejected"] += 1
                continue

            def normalize(v):
                if isinstance(v, list):
                    return sorted([str(x).lower().strip() for x in v])
                return str(v).lower().strip()

            if normalize(v1) == normalize(v2):
                # 一致 → 采纳，取较高 confidence
                c1 = p1_dim.get(field, {}).get("confidence", 0.5)
                c2 = p2_dim.get(field, {}).get("confidence", 0.5)
                evidence = p1_dim.get(field, {}).get("evidence", [])
                verified[dim_name][field] = {
                    "value": v1,
                    "confidence": max(c1, c2),
                    "evidence": evidence
                }
                stats["consistent"] += 1
            else:
                stats["rejected"] += 1

    return verified, stats


def extract_profile() -> dict:
    """
    主函数: 双验证 Profile 提取。

    返回:
        dict: {"num_consistent_fields": int, "num_total": int,
               "profile_path": str}
    """
    # ---- 加载 captions 时间线文本 ----
    captions_text, captions_dict = load_captions_text()
    clip_ids = list(captions_dict.keys())

    # ================================================================
    # 构建 Profile 提取 prompt
    # 论文对应: Table 1 - Structured prompt for profile extraction
    # ================================================================
    system_msg = STAGE3_SYSTEM

    # ---- 动态生成 schema 说明 ----
    schema_lines = ["Extract the following preference dimensions:"]
    for dim_name, dim_conf in config.PREFERENCE_DIMENSIONS.items():
        fields_str = ", ".join(dim_conf["fields"])
        schema_lines.append(
            f"  {dim_name}: {{{fields_str}}} — {dim_conf['description']}"
        )
    schema_text = "\n".join(schema_lines)

    # ================================================================
    # Pass 1: 全量 captions
    # ================================================================
    print("  Pass 1: Oracle 看全部 captions 填充 schema...")
    user_msg_1 = STAGE3_PASS1_USER.format(
        schema_text=schema_text, clip_ids=clip_ids, captions_text=captions_text)
    raw_1 = call_oracle(system_msg, user_msg_1)
    profile_1 = parse_profile_json(raw_1)
    n_fields_1 = sum(
        len([f for f in fields.values() if f.get("value") is not None])
        for fields in profile_1.values()
    )
    print(f"  Pass 1 完成: {n_fields_1} 个字段被填充")

    # ================================================================
    # Pass 2: 只看 evidence clips（关键！闭卷验证）
    # 论文对应: "第二次：去掉 LLM 上次填的内容，只给它证据 clip 列表"
    # ================================================================
    print("  Pass 2: Oracle 只看 evidence clips（闭卷验证）...")

    # ---- 收集 Pass 1 中所有被引用的 evidence clip IDs ----
    cited_clip_ids = set()
    for dim_name, fields in profile_1.items():
        for field_name, field_data in fields.items():
            if field_data and field_data.get("value") is not None:
                for ev in field_data.get("evidence", []):
                    if isinstance(ev, dict) and "clip_id" in ev:
                        cited_clip_ids.add(ev["clip_id"])
                    elif isinstance(ev, str):
                        # 字符串格式: "DAY1_A1_JAKE_17133000: ..." → 提取 clip_id
                        clip_id = ev.split(":")[0].strip()
                        if clip_id:
                            cited_clip_ids.add(clip_id)

    # ---- 只给 Oracle 看这些被引用 clip 的 caption ----
    evidence_text_lines = []
    for cid in sorted(cited_clip_ids):
        if cid in captions_dict:
            evidence_text_lines.append(
                f"[{cid}] {captions_dict[cid]['description']}"
            )
    evidence_text = "\n\n".join(evidence_text_lines)

    if not evidence_text.strip():
        print("  ⚠ Pass 1 未引用任何 evidence clips，跳过 Pass 2")
        profile_2 = {}
    else:
        user_msg_2 = STAGE3_PASS2_USER.format(
            schema_text=schema_text, evidence_text=evidence_text)
        raw_2 = call_oracle(system_msg, user_msg_2)
        profile_2 = parse_profile_json(raw_2)

    # ================================================================
    # 双验证一致性检查
    # ================================================================
    verified_profile, stats = verify_fields_consistency(profile_1, profile_2)

    print(f"  验证结果: {stats['consistent']}/{stats['total_fields']} 一致, "
          f"{stats['rejected']} 被拒绝")

    # ---- 保存 verified profile ----
    profile_output = {
        "individual_id": "subject_001",
        "profile": verified_profile,
        "verification_stats": stats,
        "pass1_raw": profile_1,
        "pass2_raw": profile_2,
        "model": config.LLM_MODEL
    }
    profile_path = os.path.join(config.PROFILE_DIR, "M_star.json")
    with open(profile_path, "w") as f:
        json.dump(profile_output, f, indent=2, ensure_ascii=False)

    return {
        "num_consistent_fields": stats["consistent"],
        "num_total_fields": stats["total_fields"],
        "rejected": stats["rejected"],
        "profile_path": profile_path
    }


if __name__ == "__main__":
    extract_profile()
