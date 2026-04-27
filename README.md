# Banking Intent Detection — Fine-tuning with Unsloth

Fine-tune **Llama-3.2-1B-Instruct** (4-bit QLoRA) on a 45-intent subset of
the **BANKING77** dataset using [Unsloth](https://github.com/unslothai/unsloth).

## Project Structure
```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # data download, sampling, cleaning, split
│   ├── train.py             # fine-tuning pipeline
│   └── inference.py         # standalone IntentClassification class
├── configs/
│   ├── train.yaml           # all training hyperparameters
│   └── inference.yaml       # checkpoint & tokenizer paths
├── sample_data/
│   ├── train.csv            # sampled training split
│   └── test.csv             # sampled test split
├── train.sh                 # one-shot training script
├── inference.sh             # one-shot inference script
├── requirements.txt
└── README.md
```

## Environment Setup
```bash
# Recommended: Python 3.10+, CUDA 11.8+
pip install -r requirements.txt
```

## Training
```bash
bash train.sh
# or step by step:
python scripts/preprocess_data.py --config configs/train.yaml
python scripts/train.py           --config configs/train.yaml
```

### Key Hyperparameters
| Parameter | Value |
|---|---|
| Base model | Llama-3.2-1B-Instruct |
| Quantisation | 4-bit (QLoRA) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective 16) |
| Epochs | 3 |
| Optimiser | AdamW 8-bit |
| LR scheduler | Cosine |
| Intents | 45 of 77 |

## Inference
```bash
# Single message
bash inference.sh "My card was declined at the ATM."

# Or in Python:
from scripts.inference import IntentClassification
clf = IntentClassification("configs/inference.yaml")
print(clf("I was charged twice for one transaction."))
```

## Results
| Metric | Value |
|---|---|
| Test accuracy | _update after training_ |

## Demo Video
> _Upload to Google Drive and paste link here._
