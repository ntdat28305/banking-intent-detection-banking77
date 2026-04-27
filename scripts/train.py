"""
train.py  —  Fine-tune Llama-3.2-1B-Instruct on BANKING77 subset.
Usage:
    python scripts/train.py --config configs/train.yaml
"""
import argparse, json, os
import yaml
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from preprocess_data import main as preprocess, TRAIN_TEMPLATE, clean_text


def train(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load model & tokenizer ───────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = cfg["model_name"],
        max_seq_length= cfg["max_seq_length"],
        load_in_4bit  = cfg["load_in_4bit"],
        dtype         = None,
    )

    # ── Preprocess data with real EOS ────────────────────────
    eos = tokenizer.eos_token
    train_df, test_df, label_map, full_df = preprocess(cfg, eos_token=eos)

    # Rebuild full formatted column with correct EOS
    def fmt(row):
        return TRAIN_TEMPLATE.format(
            message=clean_text(row["original_text"]),
            label=row["label_name"]
        ) + eos

    full_df["text"] = full_df.apply(fmt, axis=1)
    train_df = full_df.sample(frac=1-cfg["test_size"], random_state=cfg["seed"])
    # Simple re-split for consistent subset
    train_df, test_df = train_test_split(
        full_df, test_size=cfg["test_size"],
        random_state=cfg["seed"], stratify=full_df["label"],
    )

    # ── LoRA ─────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_train = HFDataset.from_pandas(train_df[["text"]])

    # ── Training args ─────────────────────────────────────────
    args = TrainingArguments(
        output_dir = cfg.get("output_dir", "./ckpts"),
        num_train_epochs = cfg["num_epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        gradient_accumulation_steps = cfg["gradient_accumulation_steps"],
        learning_rate = cfg["learning_rate"],
        weight_decay  = cfg["weight_decay"],
        warmup_ratio = cfg.get("warmup_ratio", 0.05),
        optim  = cfg["optimizer"],
        lr_scheduler_type = cfg["lr_scheduler"],
        seed  = cfg["seed"],
        fp16=False, bf16=False,
        logging_steps = cfg.get("logging_steps", 10),
        save_strategy  = "epoch",
        save_total_limit = 2,
        report_to = "none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=hf_train, args=args,
        formatting_func=lambda ex: ex["text"],
        max_seq_length=cfg["max_seq_length"],
        packing=False,
    )

    print("Training …")
    stats = trainer.train()
    print(f"Done — loss: {stats.training_loss:.4f}")

    # ── Save ─────────────────────────────────────────────────
    ckpt = cfg["checkpoint_save_path"]
    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    import shutil
    shutil.copy(f"{cfg['output_dir']}/label_map.json", f"{ckpt}/label_map.json")
    print(f"✓ Checkpoint saved to {ckpt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    args = p.parse_args()
    train(args.config)
