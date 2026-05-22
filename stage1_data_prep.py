#!/usr/bin/env python3
"""
Stage 1: EgoLife 数据准备
==========================
从 HuggingFace lmms-lab/EgoLife 下载指定个体和天的视频片段、标注数据，
抽帧并构建帧索引，将 SRT caption 合并为 clip-level 文字描述。

输入: 无（从 HuggingFace 下载）
输出:
  egolife_data/
  └── A1_JAKE/
      └── DAY1/
          ├── videos/          # 原始 30s MP4 片段
          ├── frames/          # 抽帧图片
          ├── dense_caption/   # SRT 精细动作描述
          ├── transcript/      # SRT 对话转录
          ├── clips_metadata.json   # clip 元信息 + 合并后的描述
          └── frame_index.json      # 帧索引（EgoMemReason 格式）

使用方式:
    python stage1_data_prep.py
    python stage1_data_prep.py --identity A1_JAKE --day DAY1
    python stage1_data_prep.py --download-only  # 只下载，不处理
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import config


# ================================================================
# SRT 解析
# ================================================================

def parse_srt_timestamp(ts: str) -> float:
    """将 SRT 时间戳 'HH:MM:SS,mmm' 转为秒数."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts)
    if not match:
        raise ValueError(f"无效的 SRT 时间戳: {ts}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def load_srt(srt_path: str) -> List[dict]:
    """
    解析 SRT 文件，返回 [{start_sec, end_sec, text}] 列表。
    EgoLife 的 DenseCaption SRT 是单行中文，Transcript SRT 是双语（中文+英文）。
    """
    entries = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # SRT 块之间用空行分隔
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # 第一行是序号（跳过）
        # 第二行是时间戳
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1]
        )
        if not ts_match:
            continue

        start_sec = parse_srt_timestamp(ts_match.group(1))
        end_sec = parse_srt_timestamp(ts_match.group(2))

        # 剩余行是文本内容
        text_lines = []
        for line in lines[2:]:
            line = line.strip()
            if line:
                text_lines.append(line)

        entries.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": " ".join(text_lines)
        })

    return entries


# ================================================================
# 视频文件名解析
# ================================================================

def parse_video_timestamp(filename: str) -> Tuple[str, float]:
    """
    从 EgoLife 视频文件名解析时间和时长。

    文件名格式: DAY1_A1_JAKE_11094208.mp4
    时间戳 11094208 = HHMMSSss → 11:09:42.08

    返回: (identity, day, absolute_seconds_from_day_start)
    """
    # 匹配模式: DAY{day}_{identity}_{timestamp}.mp4
    match = re.match(r"DAY(\d+)_(.+)_(\d{8,10})\.mp4", filename)
    if not match:
        raise ValueError(f"无法解析视频文件名: {filename}")

    day_num = int(match.group(1))
    identity = match.group(2)
    ts_str = match.group(3)

    # 解析时间戳 HHMMSSss 或 HHMMSSff
    if len(ts_str) == 8:
        hh = int(ts_str[0:2])
        mm = int(ts_str[2:4])
        ss = int(ts_str[4:6])
        ff = int(ts_str[6:8])
        secs = hh * 3600 + mm * 60 + ss + ff / 100.0
    elif len(ts_str) == 10:
        hh = int(ts_str[0:2])
        mm = int(ts_str[2:4])
        ss = int(ts_str[4:6])
        ms = int(ts_str[6:10])
        secs = hh * 3600 + mm * 60 + ss + ms / 10000.0
    else:
        raise ValueError(f"无法解析时间戳: {ts_str}")

    return identity, f"DAY{day_num}", secs


def format_time_hhmmss(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ================================================================
# HuggingFace 下载
# ================================================================

def download_file_hf(
    repo: str,
    path_in_repo: str,
    local_path: str,
    max_retries: int = 3
) -> bool:
    """
    从 HuggingFace 下载单个文件，支持断点续传。
    返回 True 表示成功。
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 如果文件已存在且大小 > 0，跳过
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"    ⏭ 已存在，跳过: {os.path.basename(local_path)}")
        return True

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path_in_repo}?download=true"

    for attempt in range(max_retries):
        try:
            import requests
            print(f"    ⬇ 下载: {os.path.basename(local_path)} ", end="")
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r    ⬇ 下载: {os.path.basename(local_path)} {pct}%", end="")
            print(f" ✓ ({downloaded / 1024 / 1024:.1f} MB)")
            return True

        except Exception as e:
            print(f"\n    ⚠ 尝试 {attempt+1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return False


def list_hf_dir(repo: str, path_in_repo: str) -> List[str]:
    """列出 HuggingFace 仓库中某个目录下的所有文件名。"""
    try:
        import requests
        url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{path_in_repo}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [item["path"] for item in data if item["type"] == "file"]
    except Exception as e:
        print(f"    ⚠ 无法列出目录 {path_in_repo}: {e}")
        return []


# ================================================================
# 帧抽取
# ================================================================

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: int = 1,
    max_side: int = 512
) -> List[str]:
    """
    从视频中按 FPS 抽帧，缩放到 max_side。
    返回抽取的帧路径列表。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 检查 ffmpeg 是否可用
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("需要 ffmpeg，请先安装: brew install ffmpeg")

    basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_pattern = os.path.join(output_dir, f"{basename}_%04d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps},scale='min({max_side},iw):min({max_side},ih):force_original_aspect_ratio=decrease'",
        "-q:v", "3",               # JPEG 质量 (2-5, 越小越好)
        "-loglevel", "error",
        frame_pattern
    ]
    subprocess.run(cmd, check=True)

    # 收集生成的帧
    frame_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith(basename) and f.endswith(".jpg")
    ])
    return [os.path.join(output_dir, f) for f in frame_files]


# ================================================================
# SRT → Clip 对齐
# ================================================================

def align_srt_to_clip(
    srt_entries: List[dict],
    clip_start_sec: float,
    clip_end_sec: float,
    srt_base_offset: float = 0.0
) -> str:
    """
    将 SRT 条目筛选出属于某个 clip 时间窗口的，合并为一段描述。

    参数:
      srt_entries:    SRT 条目列表 [{start_sec, end_sec, text}]
      clip_start_sec: clip 开始的绝对秒数（相对当天0点）
      clip_end_sec:   clip 结束的绝对秒数
      srt_base_offset: SRT 文件的基础偏移（如 SRT 从 11:00:00 开始，则 offset = 39600）

    返回: 合并后的文本描述
    """
    clip_texts = []
    for entry in srt_entries:
        abs_start = srt_base_offset + entry["start_sec"]
        abs_end = srt_base_offset + entry["end_sec"]

        # 检查与 clip 时间窗口的重叠
        if abs_end >= clip_start_sec and abs_start <= clip_end_sec:
            clip_texts.append(entry["text"])

    return " ".join(clip_texts)


# ================================================================
# 主流程
# ================================================================

def process_identity_day(
    identity: str,
    day: str,
    download_only: bool = False,
    skip_download: bool = False
) -> dict:
    """
    处理一个个体一天的数据：下载 → 抽帧 → 建索引 → 合并 caption。

    参数:
      download_only: True = 只下载，不处理
      skip_download:  True = 跳过下载（数据已存在）

    返回统计信息 dict。
    """
    stats = {"videos": 0, "frames": 0, "captions": 0, "skipped": 0}

    # ---- 目录结构 ----
    base = os.path.join(config.EGOLIFE_DATA_DIR, identity, day)
    video_dir = os.path.join(base, "videos")
    frame_dir = os.path.join(base, "frames")
    densecap_dir = os.path.join(base, "dense_caption")
    trans_dir = os.path.join(base, "transcript")

    for d in [video_dir, frame_dir, densecap_dir, trans_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Step 1: 列出 HF 上的文件 ----
    repo = config.EGOLIFE_HF_REPO
    hf_video_dir = f"{identity}/{day}"
    hf_densecap_dir = f"{config.EGOLIFE_DENSE_CAPTION_DIR}/{identity}/{day}"
    hf_trans_dir = f"{config.EGOLIFE_TRANSCRIPT_DIR}/{identity}/{day}"

    print(f"\n{'='*60}")
    print(f"  处理 {identity} / {day}")
    print(f"{'='*60}")

    if skip_download:
        print(f"\n  ⏭ 跳过下载（数据已就绪）")
        # 从本地扫描已有文件
        video_files_local = sorted([
            f for f in os.listdir(video_dir) if f.endswith(".mp4")
        ])
        densecap_files_local = sorted([
            f for f in os.listdir(densecap_dir) if f.endswith(".srt")
        ])
        trans_files_local = sorted([
            f for f in os.listdir(trans_dir) if f.endswith(".srt")
        ])
        print(f"  本地视频: {len(video_files_local)} 个")
        print(f"  本地 DenseCaption: {len(densecap_files_local)} 个")
        print(f"  本地 Transcript: {len(trans_files_local)} 个")

        if not video_files_local:
            print("  ⚠ 无本地视频文件，请先下载或检查 EGOLIFE_DATA_DIR")
            return stats

        video_files = [f"{identity}/{day}/{f}" for f in video_files_local]
        densecap_files = [
            f"{config.EGOLIFE_DENSE_CAPTION_DIR}/{identity}/{day}/{f}"
            for f in densecap_files_local
        ]
        trans_files = [
            f"{config.EGOLIFE_TRANSCRIPT_DIR}/{identity}/{day}/{f}"
            for f in trans_files_local
        ]
        stats["videos"] = len(video_files_local)
        stats["captions"] = len(densecap_files_local) + len(trans_files_local)
    else:
        print(f"\n[1/5] 扫描 HuggingFace 文件列表...")
        video_files = list_hf_dir(repo, hf_video_dir)
        densecap_files = list_hf_dir(repo, hf_densecap_dir)
        trans_files = list_hf_dir(repo, hf_trans_dir)

        print(f"  视频片段: {len(video_files)} 个")
        print(f"  DenseCaption SRT: {len(densecap_files)} 个")
        print(f"  Transcript SRT: {len(trans_files)} 个")

        if not video_files:
            print("  ⚠ 未找到视频文件，请检查 identity/day 是否正确")
            return stats

        # ---- Step 2: 下载视频 ----
        print(f"\n[2/5] 下载视频片段 ({len(video_files)} 个)...")
        for vf in sorted(video_files):
            fname = vf.split("/")[-1]
            local_path = os.path.join(video_dir, fname)
            if download_file_hf(repo, vf, local_path):
                stats["videos"] += 1

        # ---- Step 3: 下载 SRT 标注 ----
        print(f"\n[3/5] 下载 SRT 标注...")

        print("  DenseCaption:")
        for sf in sorted(densecap_files):
            fname = sf.split("/")[-1]
            local_path = os.path.join(densecap_dir, fname)
            if download_file_hf(repo, sf, local_path):
                stats["captions"] += 1

        print("  Transcript:")
        for sf in sorted(trans_files):
            fname = sf.split("/")[-1]
            local_path = os.path.join(trans_dir, fname)
            if download_file_hf(repo, sf, local_path):
                pass

    if download_only and not skip_download:
        print(f"\n  ✓ 下载完成! 文件位置: {base}")
        print(f"    视频: {stats['videos']} | SRT: {stats['captions']}")
        return stats

    # ---- Step 4/5: 抽帧 + 建索引 + 合并 caption ----
    step_num = "4/5" if not skip_download else "1/2"
    print(f"\n[{step_num}] 抽帧 + 对齐 caption...")

    # 先加载所有 SRT 数据到内存
    all_densecap: Dict[str, List[dict]] = {}  # {srt_basename: [entries]}
    all_transcript: Dict[str, List[dict]] = {}

    for sf in sorted(densecap_files):
        fname = sf.split("/")[-1]
        local_path = os.path.join(densecap_dir, fname)
        if os.path.exists(local_path):
            srt_key = os.path.splitext(fname)[0]
            all_densecap[srt_key] = load_srt(local_path)

    for sf in sorted(trans_files):
        fname = sf.split("/")[-1]
        local_path = os.path.join(trans_dir, fname)
        if os.path.exists(local_path):
            srt_key = os.path.splitext(fname)[0]
            all_transcript[srt_key] = load_srt(local_path)

    # 为每个 SRT 文件提取 base offset
    # SRT 文件名: A1_JAKE_DAY1_11000000 → base hour = 11:00:00
    def get_srt_offset(srt_basename: str) -> float:
        parts = srt_basename.split("_")
        if len(parts) >= 3:
            ts_str = parts[-1]  # e.g., "11000000"
            hh = int(ts_str[0:2])
            mm = int(ts_str[2:4])
            return hh * 3600 + mm * 60
        return 0.0

    # 处理每个视频
    clips_metadata = []
    frame_index = []
    video_list = sorted(os.listdir(video_dir))

    for vi, vf in enumerate(video_list):
        if not vf.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, vf)
        try:
            ident, day_label, clip_start_sec = parse_video_timestamp(vf)
        except ValueError as e:
            print(f"    ⚠ 跳过 {vf}: {e}")
            continue

        clip_end_sec = clip_start_sec + config.CLIP_DURATION

        # ---- 抽帧 ----
        clip_frame_dir = os.path.join(frame_dir, os.path.splitext(vf)[0])
        try:
            frame_paths = extract_frames_from_video(
                video_path, clip_frame_dir, fps=config.FRAME_FPS
            )
        except Exception as e:
            print(f"    ⚠ 抽帧失败 {vf}: {e}")
            stats["skipped"] += 1
            continue

        stats["frames"] += len(frame_paths)

        # ---- 构建帧索引条目 ----
        for fp in frame_paths:
            frame_index.append({
                "identity": identity,
                "day": day,
                "time": int(clip_start_sec * 100),  # 百分秒格式，对齐 EgoMemReason
                "path": os.path.abspath(fp),
                "clip": vf,
                "frame_file": os.path.basename(fp)
            })

        # ---- 对齐 SRT caption ----
        caption_parts = []
        clip_start_abs = clip_start_sec

        # DenseCaption: 查找覆盖此 clip 时间范围的 SRT 文件
        for srt_key, entries in all_densecap.items():
            offset = get_srt_offset(srt_key)
            srt_end = offset + max(e["end_sec"] for e in entries) if entries else offset
            # 检查这个 SRT 文件是否覆盖 clip 的时间
            if offset <= clip_start_abs + config.CLIP_DURATION and srt_end >= clip_start_abs:
                text = align_srt_to_clip(entries, clip_start_abs, clip_end_sec, offset)
                if text:
                    caption_parts.append(f"[DenseCaption] {text}")

        # Transcript
        for srt_key, entries in all_transcript.items():
            offset = get_srt_offset(srt_key)
            srt_end = offset + max(e["end_sec"] for e in entries) if entries else offset
            if offset <= clip_start_abs + config.CLIP_DURATION and srt_end >= clip_start_abs:
                text = align_srt_to_clip(entries, clip_start_abs, clip_end_sec, offset)
                if text:
                    caption_parts.append(f"[Transcript] {text}")

        combined_caption = " | ".join(caption_parts) if caption_parts else ""

        # ---- Clip 元信息 ----
        clip_meta = {
            "clip_id": os.path.splitext(vf)[0],
            "identity": identity,
            "day": day,
            "start_time_sec": clip_start_sec,
            "start_time_str": format_time_hhmmss(clip_start_sec),
            "end_time_sec": clip_end_sec,
            "end_time_str": format_time_hhmmss(clip_end_sec),
            "video_path": os.path.abspath(video_path),
            "frames_dir": clip_frame_dir,
            "num_frames": len(frame_paths),
            "dense_caption": combined_caption,
            "has_dense_caption": len(caption_parts) > 0,
        }
        clips_metadata.append(clip_meta)

        # 进度
        if (vi + 1) % 10 == 0 or vi == len(video_list) - 1:
            has_cap = sum(1 for c in clips_metadata if c["has_dense_caption"])
            print(f"    [{vi+1}/{len(video_list)}] clip 处理完成 "
                  f"({has_cap} 个有 caption)")

    # ---- 写入元数据文件 ----
    step_num2 = "5/5" if not skip_download else "2/2"
    print(f"\n[{step_num2}] 写入元数据文件...")

    # clips_metadata.json
    clips_meta_path = os.path.join(base, "clips_metadata.json")
    # 按时间排序
    clips_metadata.sort(key=lambda c: c["start_time_sec"])
    with open(clips_meta_path, "w", encoding="utf-8") as f:
        json.dump(clips_metadata, f, indent=2, ensure_ascii=False)

    # frame_index.json — 对齐 EgoMemReason 格式
    frame_index_path = os.path.join(base, "frame_index.json")
    with open(frame_index_path, "w", encoding="utf-8") as f:
        json.dump({identity: frame_index}, f, indent=2, ensure_ascii=False)

    # 总索引文件（合并所有 identity）
    global_index_path = os.path.join(config.EGOLIFE_DATA_DIR, "egolife_frames_index.json")
    global_index = {}
    if os.path.exists(global_index_path):
        with open(global_index_path) as f:
            global_index = json.load(f)
    global_index[identity] = frame_index
    with open(global_index_path, "w", encoding="utf-8") as f:
        json.dump(global_index, f, indent=2, ensure_ascii=False)

    # 统计信息
    print(f"\n  ✓ 完成 {identity}/{day}:")
    print(f"    视频: {stats['videos']} | 总帧数: {stats['frames']}")
    print(f"    SRT caption 文件: {stats['captions']}")
    print(f"    Clips 有 caption: {sum(1 for c in clips_metadata if c['has_dense_caption'])}/{len(clips_metadata)}")
    print(f"    Clips metadata: {clips_meta_path}")
    print(f"    Frame index:    {frame_index_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="EgoLife 数据准备")
    parser.add_argument("--identity", type=str, default=None,
                        help="个体标识 (如 A1_JAKE)，默认处理 config 中配置的所有个体")
    parser.add_argument("--day", type=str, default=None,
                        help="天标识 (如 DAY1)，默认处理 config 中配置的所有天")
    parser.add_argument("--download-only", action="store_true",
                        help="只下载文件，不抽帧不处理")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过下载（数据已存在时使用）")
    args = parser.parse_args()

    identities = [args.identity] if args.identity else config.EGOLIFE_IDENTITIES
    days = [args.day] if args.day else config.EGOLIFE_DAYS

    print("=" * 60)
    print("  EgoMind Stage 1 — EgoLife 数据准备")
    print("=" * 60)
    print(f"  数据根目录: {config.EGOLIFE_DATA_DIR}")
    print(f"  个体: {identities}")
    print(f"  天: {days}")
    print(f"  模式: {'仅下载' if args.download_only else '下载+抽帧+建索引'}")
    print(f"  抽帧 FPS: {config.FRAME_FPS}")
    print(f"  切片时长: {config.CLIP_DURATION}s")
    print("=" * 60)

    # 检查依赖
    if not args.download_only and not args.skip_download:
        try:
            import requests
        except ImportError:
            print("需要 requests 库: pip install requests")
            sys.exit(1)

        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("需要 ffmpeg: brew install ffmpeg")
            sys.exit(1)

    total_start = time.time()
    total_stats = {"videos": 0, "frames": 0, "captions": 0, "skipped": 0}

    for identity in identities:
        for day in days:
            stats = process_identity_day(
                identity, day,
                download_only=args.download_only,
                skip_download=args.skip_download
            )
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Stage 1 完成! 总耗时: {elapsed:.1f}s")
    print(f"  视频: {total_stats['videos']} | 帧: {total_stats['frames']}")
    print(f"  SRT: {total_stats['captions']} | 跳过: {total_stats['skipped']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
