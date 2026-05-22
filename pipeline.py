#!/usr/bin/env python3
"""
EgoMind 数据实验 — 主控脚本
=============================
从 EgoLife 第一人称视频出发，逐步生成可用于训练 EgoMind 的 inquiry trace 数据。

流程:
    Stage 1(数据准备) → 2(双层caption) → 3(profile) → 4(三步QA)
    → 5(过滤) → 6(trace合成) → 7(验证+输出)

使用方式：
    python pipeline.py                     # 从头运行
    python pipeline.py --dry-run           # 只看不跑
    python pipeline.py --stage 3           # 从 Stage 3 恢复
    python pipeline.py --skip-download     # Stage 1 跳过下载
"""

import argparse
import sys
import time

import config
from stage1_data_prep import process_identity_day
from stage2_caption import generate_double_layer_captions
from stage3_profile import extract_profile
from stage4_qa import generate_questions
from stage5_filter import run_filter_pipeline
from stage6_trace import synthesize_traces
from stage7_verify import verify_and_score


def stage1_run(skip_download: bool = False) -> dict:
    total_stats = {"videos": 0, "frames": 0, "clips": 0}
    for identity in config.EGOLIFE_IDENTITIES:
        for day in config.EGOLIFE_DAYS:
            stats = process_identity_day(identity, day, skip_download=skip_download)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)
    return total_stats


STAGES = {
    1: ("数据准备",                   stage1_run),
    2: ("双层 Caption 生成",          generate_double_layer_captions),
    3: ("提取偏好画像 Profile",       extract_profile),
    4: ("三步问题生成",               generate_questions),
    5: ("自动过滤",                   run_filter_pipeline),
    6: ("合成 Inquiry Traces",        synthesize_traces),
    7: ("验证 + Sparsity 评分 + 输出", verify_and_score),
}


def run_pipeline(start_stage: int = None, dry_run: bool = False,
                 skip_download: bool = False):
    if start_stage is None:
        start_stage = 1

    print("=" * 60)
    print("  EgoMind 数据实验 Pipeline")
    print("=" * 60)
    print(f"  起始阶段: Stage {start_stage}")
    print(f"  个体:     {config.EGOLIFE_IDENTITIES}")
    print(f"  天:       {config.EGOLIFE_DAYS}")
    print(f"  数据目录: {config.EGOLIFE_DATA_DIR}")
    print(f"  Caption VLM: {config.VLM_MODEL}")
    print(f"  输入方式:    video_url (30s mp4)")
    print(f"  盲测模型:    {config.BLIND_TEST_MODELS}")
    print(f"  Text LLM: {config.TEXT_LLM_MODEL}")
    print(f"  Clip 时长: {config.CLIP_DURATION}s")
    print("=" * 60)

    total_start = time.time()

    for stage_id in range(start_stage, max(STAGES.keys()) + 1):
        if stage_id not in STAGES:
            continue

        name, func = STAGES[stage_id]

        print(f"\n{'─' * 50}")
        print(f"  Stage {stage_id}: {name}")
        print(f"{'─' * 50}")

        if dry_run:
            print(f"  [DRY-RUN] 将调用 {func.__module__}.{func.__name__}()")
            continue

        stage_start = time.time()
        try:
            if stage_id == 1:
                result = func(skip_download=skip_download)
            else:
                result = func()
            elapsed = time.time() - stage_start
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, str) and len(v) > 80:
                        print(f"  ✓ {k}: ...{v[-60:]}")
                    else:
                        print(f"  ✓ {k}: {v}")
            print(f"  ✓ Stage {stage_id} 完成 (耗时 {elapsed:.1f}s)")
        except Exception as e:
            print(f"  ✗ Stage {stage_id} 失败: {e}")
            if not dry_run:
                print(f"  💡 修复问题后可用 'python pipeline.py --stage {stage_id}' 从此阶段恢复")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline 完成! 总耗时: {total_elapsed:.1f}s")
    print(f"  训练样本: {config.TRAINING_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EgoMind 数据实验 Pipeline")
    parser.add_argument("--stage", type=int, default=None,
                        choices=range(1, 9),
                        help="从哪个阶段开始 (1-7)，用于从中断处恢复")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印流程，不实际执行")
    parser.add_argument("--skip-download", action="store_true",
                        help="Stage 1 跳过下载")
    args = parser.parse_args()
    run_pipeline(start_stage=args.stage, dry_run=args.dry_run,
                 skip_download=args.skip_download)
