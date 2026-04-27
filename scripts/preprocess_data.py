"""
preprocess_data.py
Downloads BANKING77, samples intent subset, cleans text,
and writes train.csv / test.csv to sample_data/.
"""
import json, random, argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


TRAIN_TEMPLATE = (
    "A customer sent the following message to a bank.\n"
    "Identify the intent category precisely.\n\n"
    "Customer message: {message}\n\n"
    "Intent: {label}"
)

INFER_TEMPLATE = (
    "A customer sent the following message to a bank.\n"
    "Identify the intent category precisely.\n\n"
    "Customer message: {message}\n\n"
    "Intent:"
)


def clean_text(text: str) -> str:
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".?!":
        text += "."
    return text


def main(cfg: dict, eos_token: str = "</s>"):
    random.seed(cfg["seed"])
    raw = load_dataset(cfg["dataset_name"], revision="refs/convert/parquet")
    all_int = dict(enumerate(raw["train"].features["label"].names))
    ids = sorted(random.sample(list(all_int.keys()), cfg["num_intents"]))
    id_set = set(ids)
    old2new = {o: n for n, o in enumerate(ids)}
    label_map = {n: all_int[o] for n, o in enumerate(ids)}

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(f"{cfg['output_dir']}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    rows = []
    for split in ("train", "test"):
        for ex in raw[split]:
            if ex["label"] not in id_set:
                continue
            msg  = clean_text(ex["text"])
            name = all_int[ex["label"]]
            rows.append({
                "text" : TRAIN_TEMPLATE.format(message=msg, label=name) + eos_token,
                "label" : old2new[ex["label"]],
                "label_name" : name,
                "original_text" : ex["text"],
            })

    df = pd.DataFrame(rows)
    tr, te = train_test_split(df, test_size=cfg["test_size"],
                              random_state=cfg["seed"], stratify=df["label"])
    tr[["original_text", "label_name"]].to_csv(f"{cfg['output_dir']}/train.csv", index=False)
    te[["original_text", "label_name"]].to_csv(f"{cfg['output_dir']}/test.csv", index=False)
    print(f"Saved {len(tr)} train / {len(te)} test rows → {cfg['output_dir']}/")
    return tr, te, label_map, df


if __name__ == "__main__":
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    import os
    main(cfg)
