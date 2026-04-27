# Banking Intent Detection — Fine-tuning with Unsloth

Fine-tune **Llama-3.2-1B-Instruct** using **4-bit QLoRA** on the **BANKING77** dataset to classify customer banking intents. Built with [Unsloth](https://github.com/unslothai/unsloth) for fast and memory-efficient training.

> **Demo Video:** [Link Video Demo](https://drive.google.com/file/d/1kkcT0QjcvK_6orMYQifSLBYdnZE0nX-P/view?usp=sharing) 

---

## Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Download, sample, clean, split dataset
│   ├── train.py             # Fine-tuning pipeline with Unsloth + LoRA
│   └── inference.py         # Standalone IntentClassification class
├── configs/
│   ├── train.yaml           # All training hyperparameters
│   └── inference.yaml       # Checkpoint path & inference settings
├── sample_data/
│   ├── train.csv            # Sampled training split (80%)
│   └── test.csv             # Sampled test split (20%)
├── train.sh                 # One-shot training script
├── inference.sh             # One-shot inference script
├── requirements.txt
└── README.md
```

---

## Environment Setup

**Requirements:** Python 3.10+, CUDA 11.8+, GPU with at least 8GB VRAM (recommended: T4 on Google Colab)

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install datasets>=2.18 scikit-learn>=1.4 pandas>=2.0 pyyaml>=6.0 torch>=2.1
```

---

## Dataset

This project uses the [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset — 77 intents covering common banking customer queries.

- Both train and test splits are downloaded from HuggingFace
- Combined and re-split with stratified 80/20 ratio
- **Train samples:** 10,466 | **Test samples:** 2,617

---

## Training

**Step 1 — Preprocess data:**
```bash
python scripts/preprocess_data.py --config configs/train.yaml
```

**Step 2 — Fine-tune:**
```bash
python scripts/train.py --config configs/train.yaml
```

Or run both steps at once:
```bash
bash train.sh
```

Checkpoint will be saved to `outputs/checkpoint/` after training.

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-Instruct` |
| Quantization | 4-bit QLoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Epochs | 3 |
| Optimizer | AdamW 8-bit |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Max sequence length | 256 |
| Number of intents | 77 |

---

## Inference

**Single message via shell:**
```bash
bash inference.sh "I am still waiting on my card?"
# Output: Intent: card_arrival
```

**Multiple messages via Python:**
```python
from scripts.inference import IntentClassification

clf = IntentClassification("configs/inference.yaml")

print(clf("I am still waiting on my card?"))
# → card_arrival

print(clf("I was charged twice for one transaction."))
# → extra_charge_on_statement

print(clf("How do I change my PIN?"))
# → change_pin
```

The `IntentClassification` class follows the required interface:

```python
class IntentClassification:
    def __init__(self, model_path):
        # Loads config, tokenizer, and model checkpoint

    def __call__(self, message):
        # Returns predicted intent label string
```

`model_path` points to `configs/inference.yaml` which contains the checkpoint path and inference settings.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **92,59%** |
| Number of intents | 77 |
| Test samples | 2,617 |

---

## Demo Video

[Watch demo video](https://drive.google.com/file/d/1kkcT0QjcvK_6orMYQifSLBYdnZE0nX-P/view?usp=sharing)

The video demonstrates:
- How the inference script is executed
- Example input messages and predicted intent labels
- Final accuracy on the test set
