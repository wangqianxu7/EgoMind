"""
Stage 2: 双层 Caption 生成
===========================
借鉴 EGOMEMREASON 的 object-centric caption，加入 EgoMind 独有的 Preference Signal 标注。

与 EGOMEMREASON 对齐的输入方式：
  - EGOMEMREASON: 30s clip → 1 FPS 抽帧 → 30张图 → GPT-5
  - 本实现:      30s mp4 → video_url → Kimi K2.5（由 VLM 内部做帧采样）

双层结构：
  Layer 1 — Object-Centric Description：物体状态追踪、空间位置、人物交互
  Layer 2 — Preference Signal Annotation：对 5 维偏好画像的信号标注

使用方式：
  python stage2_caption.py --max-clips 3       # 试跑
  python stage2_caption.py --dry-run           # 只看不跑
"""

import base64
import json
import os
import sys
import time
from openai import OpenAI
import config
from prompts import STAGE2_SYSTEM, STAGE2_USER

# 输出目录
OUT_DIR = config.CAPTIONS_DIR

# 断点续跑文件
CHECKPOINT_FILE = os.path.join(OUT_DIR, "_checkpoint.json")


# ================================================================
# Utility
# ================================================================

def encode_video_b64(video_path):
    """读取 mp4 视频文件并编码为 base64 字符串。"""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed_ids):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(completed_ids), f)


# ================================================================
# 单个 Clip 处理
# ================================================================

def generate_double_layer_caption(client, clip):
    """为单个 clip 生成双层 caption。传入 30s mp4 视频给 VLM。返回 caption dict 或 None。"""
    clip_id = clip["clip_id"]
    video_path = clip.get("video_path", "")
    timestamp = clip.get("start_time_str", clip.get("timestamp", "unknown"))
    srt_text = clip.get("dense_caption", "")

    if not video_path or not os.path.exists(video_path):
        print(f"    ⚠ 无视频文件: {video_path}")
        return None

    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    # 编码视频
    video_b64 = encode_video_b64(video_path)

    # 构造请求：text + video_url
    user_msg = STAGE2_USER.format(
        clip_id=clip_id, timestamp=timestamp, srt_text=srt_text)
    content = [
        {"type": "text", "text": user_msg},
        {"type": "video_url",
         "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
    ]

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=config.VLM_MODEL,
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM},
                    {"role": "user", "content": content}
                ],
                max_tokens=config.VLM_MAX_TOKENS,
                temperature=config.VLM_TEMPERATURE,
                extra_body=config.VLM_EXTRA_BODY
            )
            raw = resp.choices[0].message.content.strip()
            # 解析 JSON（容错 markdown 包裹）
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            break
        except (json.JSONDecodeError, Exception) as e:
            if attempt == config.LLM_MAX_RETRIES - 1:
                print(f"    ✗ 解析失败: {str(e)[:100]}")
                return None
            time.sleep(2 ** attempt)

    return {
        "clip_id": clip_id,
        "timestamp": timestamp,
        "layer1": result.get("layer1", {}),
        "layer2": result.get("layer2", {}),
        "srt_reference": srt_text[:500],
        "video_size_mb": round(video_size_mb, 1),
        "input_type": "video_url",
        "model": config.VLM_MODEL
    }


# ================================================================
# 主流程
# ================================================================

def generate_double_layer_captions(max_clips=None, dry_run=False):
    """
    为所有 EgoLife clip 生成双层 caption。

    参数:
        max_clips: 最多处理多少 clip（None=用 config 设置, 0=全部）
        dry_run: 只统计不调用 API
    """
    if max_clips is None:
        max_clips = config.DOUBLE_LAYER_MAX_CLIPS

    os.makedirs(OUT_DIR, exist_ok=True)
    completed = load_checkpoint()

    # 收集所有待处理 clip
    all_clips = []
    for identity in config.EGOLIFE_IDENTITIES:
        for day in config.EGOLIFE_DAYS:
            meta_path = os.path.join(
                config.EGOLIFE_DATA_DIR, identity, day, "clips_metadata.json"
            )
            if not os.path.exists(meta_path):
                print(f"  ⚠ 未找到 {meta_path}，跳过")
                continue
            with open(meta_path) as f:
                clips = json.load(f)
            all_clips.extend(clips)

    total = len(all_clips)
    if max_clips > 0 and total > max_clips:
        step = total / max_clips
        all_clips = [all_clips[int(i * step)] for i in range(max_clips)]
        print(f"  采样 {len(all_clips)}/{total} 个 clip（max_clips={max_clips}）")
    else:
        print(f"  共 {total} 个 clip")

    if dry_run:
        print(f"  [DRY-RUN] 将处理 {len(all_clips)} 个 clip，不调用 API")
        # 粗略估算：30s mp4 ≈ 10MB → base64，以 API 实际计费为准
        est_size_mb = len(all_clips) * 10
        print(f"  预估视频数据量: ~{est_size_mb} MB (~{est_size_mb/1024:.1f} GB)")
        return {"dry_run": True, "num_clips": len(all_clips), "estimated_video_mb": est_size_mb}

    # 初始化 VLM client
    client = OpenAI(
        api_key=config.VLM_API_KEY,
        base_url=config.VLM_BASE_URL if config.VLM_BASE_URL else None
    )

    print(f"  VLM: {config.VLM_MODEL} @ {config.VLM_BASE_URL}")
    print(f"  输出: {OUT_DIR}/")
    print(f"  已断点续跑: {len(completed)} 个已完成")
    print()

    all_captions = {}
    new_count = 0
    skip_count = 0
    fail_count = 0

    for idx, clip in enumerate(all_clips):
        clip_id = clip["clip_id"]

        if clip_id in completed:
            skip_count += 1
            # 加载已有结果
            out_path = os.path.join(OUT_DIR, f"{clip_id}.json")
            if os.path.exists(out_path):
                with open(out_path) as f:
                    all_captions[clip_id] = json.load(f)
            continue

        print(f"  [{idx+1}/{len(all_clips)}] {clip_id} "
              f"({clip.get('start_time_str', '?')})...", end=" ", flush=True)

        result = generate_double_layer_caption(client, clip)
        if result:
            all_captions[clip_id] = result
            # 保存单个文件
            out_path = os.path.join(OUT_DIR, f"{clip_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            # 更新断点
            completed.add(clip_id)
            save_checkpoint(completed)
            new_count += 1
            print(f"✓ ({result.get('video_size_mb','?')}MB, "
                  f"L1 objects: {len(result['layer1'].get('objects',[]))}, "
                  f"L2 signals: {count_signals(result['layer2'])})")
        else:
            fail_count += 1
            print("✗")

        # 限流
        if config.DOUBLE_LAYER_RATE_LIMIT > 0:
            time.sleep(config.DOUBLE_LAYER_RATE_LIMIT)

    # 汇总文件
    summary_path = os.path.join(OUT_DIR, "all_double_layer_captions.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 完成: {new_count} 新生成, {skip_count} 跳过, {fail_count} 失败")
    print(f"  汇总: {summary_path}")
    return {
        "num_total": len(all_clips),
        "num_new": new_count,
        "num_skipped": skip_count,
        "num_failed": fail_count,
        "captions_dir": OUT_DIR,
        "summary_json": summary_path
    }


def count_signals(layer2):
    """统计 Layer 2 中非空 signal 数量。"""
    count = 0
    for dim, fields in layer2.items():
        if isinstance(fields, dict):
            for v in fields.values():
                if v is not None and v != [] and v != "":
                    count += 1
    return count


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: 双层 Caption 生成")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="最多处理 clip 数（0=全部）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计，不调用 API")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 2: 双层 Caption 生成")
    print("=" * 60)

    generate_double_layer_captions(max_clips=args.max_clips, dry_run=args.dry_run)
