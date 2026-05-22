# EgoMind: Reasoning over Individual Behavioral Preferences from Egocentric Long Videos

**Wen Wang**  
Shenzhen Graduate School, Peking University  
wwang25@stu.pku.edu.cn

---

## Abstract

Modeling individual behavioral preferences from egocentric long videos is a foundational capability for personalized AI assistants. Existing solutions are limited in two ways: frozen multimodal large language models (MLLMs) reason via ungrounded textual chains without accumulating user-specific knowledge across sessions, while retrieval-augmented approaches perform single-pass evidence lookup without iterative refinement — brittle precisely when preferences are sparse, dispersed, or implicit. We argue that preference modeling should be reframed as a post-training problem: the MLLM itself should learn to ground its reasoning in spatiotemporal evidence and to operate on a persistent preference memory as an explicit part of its reasoning trace. We present EgoMind, a framework that teaches an MLLM to observe, remember, and infer — identifying behavior-relevant evidence, reading from and writing to a structured per-individual preference memory, and producing preference-grounded predictions through multi-step deduction. We construct a large-scale dataset of preference-annotated reasoning traces spanning mobility, temporal, spatial, social, and consumption dimensions, and design a two-phase curriculum: progressive supervised fine-tuning followed by reinforcement learning with verifiable rewards that jointly optimizes evidence grounding, memory operations, and prediction accuracy. Experiments show EgoMind substantially outperforms both retrieval-augmented and post-trained MLLM baselines, with the largest gains on long-range, cross-session preference reasoning. These results suggest that turning memory operations into a learnable behavior, rather than external scaffolding around a frozen model, offers a viable path toward scalable preference modeling from egocentric video.

**Keywords:** Egocentric Long Video, Personalized Preference Modeling, Multimodal Large Language Models, Memory-Augmented Reasoning

---

## Data Pipeline

The EgoPref dataset is constructed from EgoLife egocentric videos through a 7-stage pipeline:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `stage1_data_prep.py` | Download & preprocess EgoLife videos |
| 2 | `stage2_caption.py` | Double-layer video captioning via VLM (Kimi K2.5) |
| 3 | `stage3_profile.py` | Preference profile extraction with dual-pass verification |
| 4 | `stage4_qa.py` | Three-step question generation with memory-confusion distractors |
| 5 | `stage5_filter.py` | Automatic filtering (blind-test + temporal span) |
| 6 | `stage6_trace.py` | Multi-step inquiry trace synthesis |
| 7 | `stage7_verify.py` | Verification, sparsity scoring, and SFT/RL split |

Run the full pipeline:

```bash
pip install openai
python pipeline.py --skip-download    # with pre-downloaded data
python pipeline.py --dry-run          # preview without API calls
```

Configuration is managed in `config.py` and prompt templates in `prompts.py`.

## Requirements

- Python 3.10+
- OpenAI-compatible API endpoints (Kimi K2.5 for VLM, DeepSeek-Chat for text)
- ffmpeg (for frame extraction)

## License

This project is for research purposes. See paper for details.
