"""
EgoMind 数据实验 — 全局配置
=============================
控制整个 pipeline 的所有参数：路径、LLM 设置、处理参数。
修改此文件即可调整实验行为，无需改动各阶段脚本。

基于 HuggingFace EgoLife 数据集：多人多天第一人称视频。
"""

import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 各阶段输出子目录
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
CAPTIONS_DIR = os.path.join(OUTPUT_DIR, "captions")    # 双层 caption 输出
PROFILE_DIR = os.path.join(OUTPUT_DIR, "profile")       # M*_P 偏好画像
TRACES_DIR = os.path.join(OUTPUT_DIR, "traces")          # inquiry traces
TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")      # 训练样本
QA_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "qa_pairs.json")

# ============================================================
# EgoLife 数据配置
# ============================================================
EGOLIFE_DATA_DIR = os.environ.get(
    "EGOLIFE_DATA_DIR",
    os.path.join(BASE_DIR, "egolife_data")
)

EGOLIFE_HF_REPO = "lmms-lab/EgoLife"

# 论文设计：6 位参与者全部用于数据处理；train/test 划分在最终样本输出时按 7:3 随机切分
EGOLIFE_IDENTITIES = ["A1_JAKE", "A2_ALICE", "A3_TASHA",
                      "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
EGOLIFE_DAYS = ["DAY1", "DAY2", "DAY3", "DAY4", "DAY5", "DAY6", "DAY7"]

EGOLIFE_DENSE_CAPTION_DIR = "EgoLifeCap/DenseCaption"
EGOLIFE_TRANSCRIPT_DIR = "EgoLifeCap/Transcript"
EGOLIFE_QA_DIR = "EgoLifeQA"

# ============================================================
# 视频预处理参数（论文 §3.2: 30s clips, SRT 提供 time-aligned audio transcript,
#   VLM 直接分析 mp4 视频，无需额外音频提取）
CLIP_DURATION = 30
FRAME_FPS = 1

# ============================================================
# LLM 配置 — 双 Endpoint 设计
# ============================================================
# VLM (Vision-Language Model): Stage 2 视频 caption
# Text LLM: Stage 3-7 纯文本推理

# ---- VLM Endpoint ----
# KEY 优先级: VLM_API_KEY > MOONSHOT_API_KEY > OPENAI_API_KEY
VLM_API_KEY = (
    os.environ.get("VLM_API_KEY") or
    os.environ.get("MOONSHOT_API_KEY") or
    os.environ.get("OPENAI_API_KEY", "")
)
VLM_BASE_URL = (
    os.environ.get("VLM_BASE_URL") or
    "https://api.moonshot.cn/v1"
)
VLM_MODEL = os.environ.get("VLM_MODEL", "kimi-k2.5")
VLM_TEMPERATURE = 0.6         # Kimi K2.5 thinking=disabled 时只允许 0.6
VLM_MAX_TOKENS = 4096
VLM_EXTRA_BODY = {"thinking": {"type": "disabled"}}

# ---- Text LLM Endpoint ----
TEXT_LLM_API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY") or
    os.environ.get("TEXT_LLM_API_KEY") or
    os.environ.get("MOONSHOT_API_KEY") or
    os.environ.get("OPENAI_API_KEY", "")
)
TEXT_LLM_BASE_URL = (
    os.environ.get("DEEPSEEK_BASE_URL") or
    os.environ.get("TEXT_LLM_BASE_URL") or
    os.environ.get("MOONSHOT_BASE_URL") or
    os.environ.get("OPENAI_BASE_URL") or
    "https://api.deepseek.com"
)
TEXT_LLM_MODEL = os.environ.get("TEXT_LLM_MODEL", "deepseek-chat")
TEXT_LLM_TEMPERATURE_ORACLE = 0.7
TEXT_LLM_TEMPERATURE_EXTRACT = 0.3
TEXT_LLM_MAX_TOKENS = 4096

# ---- 向后兼容 ----
LLM_MODEL = VLM_MODEL
LLM_TEMPERATURE_CAPTION = VLM_TEMPERATURE
LLM_TEMPERATURE_ORACLE = TEXT_LLM_TEMPERATURE_ORACLE
LLM_MAX_TOKENS = TEXT_LLM_MAX_TOKENS
LLM_MAX_RETRIES = 3
OPENAI_API_KEY = VLM_API_KEY
OPENAI_BASE_URL = VLM_BASE_URL

# ============================================================
# 偏好 Profile Schema (5 维度)
# ============================================================
PREFERENCE_DIMENSIONS = {
    "mobility": {
        "description": "How the individual moves through space",
        "fields": ["primary_mode", "avg_distance_km", "route_consistency"]
    },
    "temporal": {
        "description": "When they are active and activity patterns",
        "fields": ["peak_activity_hour", "weekend_pattern_diff"]
    },
    "spatial": {
        "description": "Where they spend time and how spaces are used",
        "fields": ["frequented_venue_types", "indoor_outdoor_ratio"]
    },
    "social": {
        "description": "Interaction patterns and social behavior",
        "fields": ["solo_ratio", "primary_interaction_partners"]
    },
    "consumption": {
        "description": "Objects, media, and services they choose",
        "fields": ["food_categories", "brand_recurrence"]
    }
}

# ============================================================
# Trace 合成参数
# ============================================================
T_MAX = 3                   # 最大 inquiry 步数
SCAN_K = 8                  # SCAN 每次检索 clip 数
ZOOM_K = 8                  # ZOOM 每次检索 clip 数
QUESTIONS_PER_FIELD = 3     # 每字段生成问题数

# ============================================================
# 验证参数
# ============================================================
MAX_REGENERATION = 3
VERIFICATION_PASS_THRESHOLD = 0.7

# ============================================================
# 双层 Caption 参数
# ============================================================
DOUBLE_LAYER_MAX_CLIPS = 0      # 0=全量
DOUBLE_LAYER_RATE_LIMIT = 0.5   # VLM 调用间隔（秒）

# ============================================================
# 三步问题生成 & 过滤参数
# ============================================================
MAX_STATEMENTS_PER_FIELD = 8
BLIND_TEST_MODELS = ["deepseek-chat"]
BLIND_TEST_THRESHOLD = 0.5
MIN_EVIDENCE_SPAN_HOURS = 1.0
QUALITY_LOOP_MAX_ITERATIONS = 3

# ============================================================
# 确保输出目录存在
# ============================================================
for d in [CLIPS_DIR, FRAMES_DIR, CAPTIONS_DIR, PROFILE_DIR, TRACES_DIR, TRAINING_DIR]:
    os.makedirs(d, exist_ok=True)
