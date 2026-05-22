"""
EgoMind 实验 — 全部 Prompt 模板
================================
集中管理 Stage 2-7 使用的所有 LLM/VLM prompt。
共 12 段模板，全部三引号直写，所见即所得。
"""

# ================================================================
# Stage 2: 双层 Caption 生成 (Kimi K2.5)
# ================================================================

STAGE2_SYSTEM = """\
You are an expert egocentric video analyst. Your task is to produce
a structured two-layer caption for a 30-second first-person video clip.

LAYER 1 — Object-Centric Description:
  Track objects, their states, locations, and interactions.
  List people present and their roles.
  Describe actions in chronological order.

LAYER 2 — Preference Signal Annotation:
  For each of the 5 preference dimensions below, note any signals
  this clip provides — even weak ones. Set value to null if no signal.
  Dimensions:
    mobility: primary_mode, route_consistency
    temporal: time_of_day, activity_type
    spatial: venue_type, indoor_outdoor
    social: interaction_partners (list names), solo_vs_group, sharing_behavior
    consumption: food_mentions, brand_mentions, media_type, objects_used

RULES:
  - Use the SRT reference as ground truth for actions and dialogue.
  - The video provides visual context (objects, people, locations).
  - Be specific: name people when visible, describe object states precisely.
  - For Layer 2, be conservative — only annotate what is clearly observed.
  - Output valid JSON only, no markdown wrapping."""

STAGE2_USER = """\
Clip: {clip_id}
Timestamp: {timestamp}
Duration: 30 seconds

=== SRT REFERENCE (ground truth actions + dialogue) ===
{srt_text}

=== VIDEO ===
Watch the 30-second egocentric video below. Analyze it frame by frame to enrich
the SRT with object states, spatial context, and visual details.

OUTPUT JSON FORMAT:
{{
  "layer1": {{
    "scene_summary": "1-2 sentence overview",
    "objects": [
      {{"name": "...", "states": ["on table", "being held"],
        "interactions": ["wearer → Katrina"]}}
    ],
    "actions": ["chronological action list"],
    "spatial": {{"location": "...", "indoor_outdoor": "...",
                "wearer_position": "..."}},
    "people": ["name1", "name2"]
  }},
  "layer2": {{
    "mobility": {{"primary_mode": null, "route_consistency": null}},
    "temporal": {{"time_of_day": "...", "activity_type": "..."}},
    "spatial": {{"venue_type": "...", "indoor_outdoor": "..."}},
    "social": {{"interaction_partners": [], "solo_vs_group": "...",
               "sharing_behavior": null}},
    "consumption": {{"food_mentions": [], "brand_mentions": [],
                    "media_type": null, "objects_used": []}}
  }}
}}"""


# ================================================================
# Stage 3: Profile 提取
# ================================================================

STAGE3_SYSTEM = """\
You are an expert behavioral analyst. Given chronological descriptions
from an individual's egocentric (first-person) video recordings,
extract a structured preference profile.

IMPORTANT RULES:
1. Each field must contain: 'value', 'confidence' (0-1 decimal),
   and 'evidence' (list of clip_ids with brief excerpts).
2. You MUST cite at least TWO different clip_ids per field.
3. If evidence is insufficient, set value to null — NEVER GUESS.
4. 'confidence' reflects evidence strength: 0.9+ for strong pattern,
   0.5-0.7 for tentative, null fields get no confidence.
5. Output valid JSON only, no markdown."""

STAGE3_PASS1_USER = """\
{schema_text}

AVAILABLE CLIPS: {clip_ids}

CHRONOLOGICAL CAPTIONS:
{captions_text}

OUTPUT: JSON with dimensions as top-level keys.
Each field is an object with 'value', 'confidence', 'evidence'."""

STAGE3_PASS2_USER = """\
{schema_text}

BELOW ARE ONLY THE RELEVANT CLIPS (no prior judgments provided).
Independently assign values to each field based solely on this evidence.

EVIDENCE CLIPS:
{evidence_text}

OUTPUT: JSON, same format as before."""


# ================================================================
# Stage 4: 三步问题生成 — Step 1: Evidence Statement 提取
# ================================================================

STAGE4_STEP1_SYSTEM = """\
You are an evidence analyst. Given a preference field and a set of clip captions,
extract factual statements that serve as evidence for this field.

Each statement must:
  - Be a single, atomic fact (one observation per statement)
  - Reference a specific clip_id and timestamp
  - Include a confidence score (0-1)
  - Be tagged with a type: 'explicit_mention', 'visual_observation',
    'behavioral_pattern', or 'dialogue'

OUTPUT: JSON with a 'statements' array."""

STAGE4_STEP1_USER = """\
PREFERENCE FIELD: {dimension}.{field}
FIELD DESCRIPTION: {description}

CAPTIONS (with clip_id, timestamp, and double-layer annotations):
{captions_text}

TASK: Extract up to {max_statements} evidence statements for {dimension}.{field}.
Each statement should cite the clip_id it comes from.

OUTPUT FORMAT:
{{"statements": [
  {{"id": "S1", "clip_id": "DAY1_A1_JAKE_13370000",
    "timestamp": "13:37:00",
    "text": "Jake explicitly says KFC when deciding lunch",
    "confidence": 0.9, "type": "dialogue"}}
]}}"""

# ================================================================
# Stage 4: 三步问题生成 — Step 2: Query 生成
# ================================================================

STAGE4_STEP2_SYSTEM = """\
You are a question designer for training AI models to reason about
human behavioral preferences from egocentric video evidence.

Your questions must require genuine evidence gathering — not simple recall.
Each question is bound to a specific TRACE PATTERN that determines
how the model should search for evidence:

TRACE PATTERNS:
  single_scan: Direct inference from a few clips
  cross_session: Compare across different time ranges
  pattern_abstraction: Abstract recurring patterns needing confirmation
  multi_entity: Track objects/people flow across clips

QUESTION TYPES:
  multiple_choice: 4 options A/B/C/D, one correct
  open_ended: Descriptive answer required

DISTRACTOR RULES (memory-confusion based):
  - temporal_confusion: correct item, wrong time
  - person_confusion: correct event, wrong person
  - frequency_illusion: once-seen presented as habit
  - recency_bias: most recent presented as default

OUTPUT: JSON with a 'questions' array."""

STAGE4_STEP2_USER = """\
PREFERENCE FIELD: {dimension}.{field}
GROUND TRUTH VALUE: {value}
CONFIDENCE: {confidence}

EVIDENCE STATEMENTS:
{statements_text}

CAPTIONS CONTEXT (first 3 relevant clips):
{captions_summary}

TASK: Generate {num_questions} questions ({question_types}), each assigned
to a trace_pattern from: {trace_options}.
For multiple_choice questions, generate memory-confusion distractors.
For open_ended questions, provide a concise ground_truth answer.

IMPORTANT: Option text must be CLEAN — NO type labels like (temporal_confusion).
The distractor type is recorded separately in distractor_types metadata.

OUTPUT FORMAT:
{{"questions": [
  {{"id": "q_NNN", "type": "multiple_choice",
    "trace_pattern": "cross_session",
    "question": "...",
    "options": ["A. 在会议室讨论项目", "B. 在餐厅吃午餐（正确）",
                "C. 在客厅看电视", "D. 在厨房准备晚餐"],
    "ground_truth": "B",
    "distractor_types": ["temporal_confusion", "person_confusion",
                          "frequency_illusion"],
    "reasoning_type": "habit_inference"}}
]}}"""


# ================================================================
# Stage 6: Inquiry Trace 合成
# ================================================================

STAGE6_SYSTEM = """\
You are simulating an ideal evidence-gathering AI agent that reasons
about individual preferences from egocentric video clips.

CRITICAL: You MUST simulate the reasoning of an agent that does NOT
know the answer in advance. Each step should reflect GENUINE evidence
discovery — not jumping to the known answer.

AVAILABLE ACTIONS:
  A_scan(k, time_filter): Sample k random clips from a time range.
  A_zoom(k, time_filter, focus): Sample k clips near a specific context.
  A_commit(answer): Declare evidence sufficient, emit final answer.

MEMORY OPERATIONS (interleave with actions):
  R_rev(field, value, confidence, evidence): Write/update memory field.
  R_rec(dimension): Retrieve a dimension of memory for reasoning.
  R_refl(): Summarize current memory state.

MEMORY STATE SCHEMA — memory_state_after MUST follow this 5-dimension structure:
  {{"mobility": {{"primary_mode": {{"value": ..., "confidence": ...}}, ...}},
   "temporal": {{"peak_activity_hour": {{"value": ..., "confidence": ...}}, ...}},
   "spatial": {{"frequented_venue_types": {{"value": ..., "confidence": ...}}, ...}},
   "social": {{"solo_ratio": {{"value": ..., "confidence": ...}}, ...}},
   "consumption": {{"food_categories": {{"value": ..., "confidence": ...}}, ...}}
  Each field is an object with 'value' and 'confidence'. Empty/unused dims = {{}}.

EACH STEP MUST CONTAIN:
  <observe>: analysis of examined clips
  <remember>: memory operations with explicit clip_id citations
  <infer>: summary of what has been learned so far
  <action>: the next action to take

RULES:
  - Maximum {t_max} steps total
  - Your examined_clips MUST include at least one evidence clip
  - Each R_rev must cite ≥1 supporting clip_id
  - Confidence reflects evidence quantity × consistency
  - A_commit only when evidence is unambiguous
  - Output valid JSON only"""

STAGE6_USER = """\
QUESTION: {question}
TYPE: {qtype}
{options_line}GROUND TRUTH (for verification only): {ground_truth}
{evidence_info}
AVAILABLE CLIPS ({n_clips} total):
{clip_info}

TASK: Generate a {t_max}-step inquiry trace showing how to
arrive at the correct answer through evidence gathering.
IMPORTANT: Your SCAN/ZOOM actions must actually DISCOVER the evidence
clips listed above. Include them in examined_clips.

OUTPUT FORMAT (JSON):
{{"trace": [
  {{"round": 1, "observe": "...",
    "action": {{"type": "A_scan", "params": {{"k": {scan_k}, "time_filter": "..."}}}},
    "examined_clips": ["clip_0001", ...],
    "memory_operations": [{{"op": "R_rev", "field": "...",
      "value": "...", "confidence": 0.7, "evidence": ["clip_0001"]}}],
    "memory_state_after": {{"consumption": {{"brand_recurrence":
      {{"value": "KFC", "confidence": 0.7}}}}}}
  }},
  ... up to {t_max} rounds, final round must be A_commit
]}}"""


# ================================================================
# Stage 5: 自动过滤 — 文本泄漏盲测
# ================================================================

STAGE5_BLIND_SYSTEM = """\
You are taking a multiple-choice quiz. You do NOT have access to any
video, images, or captions. Answer based ONLY on the question text.
If the question requires visual or temporal evidence you don't have,
respond with 'UNKNOWN'.

OUTPUT: JSON with 'answer' field (A/B/C/D or UNKNOWN) and 'confidence' (0-1)."""

STAGE5_BLIND_USER = """\
QUESTION: {q_text}
{options_line}Answer with just the option letter (A/B/C/D) or UNKNOWN."""
