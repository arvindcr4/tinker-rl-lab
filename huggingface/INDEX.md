# huggingface/ — INDEX

**Purpose:** Publish trained RL checkpoints (and datasets) to the Hugging Face Hub with generated model cards.

**Key files:**
- `upload_to_hub.py` — CLI uploader; pushes a `--model-path` checkpoint to `--repo-id`, filling `MODEL_CARD_TEMPLATE.md` with base model / method (GRPO...) / dataset / metrics.
- `MODEL_CARD_TEMPLATE.md` — model-card template rendered by the uploader.

**Find it fast:**
- to publish a checkpoint → `python huggingface/upload_to_hub.py --model-path ... --repo-id ...`
