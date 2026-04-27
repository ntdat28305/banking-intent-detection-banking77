"""
inference.py  —  Standalone intent classifier.
Usage:
    python scripts/inference.py --config configs/inference.yaml --message "I lost my card"
"""
import argparse, json, yaml, torch
from unsloth import FastLanguageModel


_INFER_TEMPLATE = (
    "A customer sent the following message to a bank.\n"
    "Identify the intent category precisely.\n\n"
    "Customer message: {message}\n\n"
    "Intent:"
)


class IntentClassification:
    """
    Banking intent classifier.
    Args:
        model_path: path to inference.yaml config file.
    """

    def __init__(self, model_path: str):
        with open(model_path) as f:
            cfg = yaml.safe_load(f)

        with open(cfg["label_map_path"]) as f:
            self.label_map = json.load(f)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name  = cfg["checkpoint_path"],
            max_seq_length= cfg.get("max_seq_length", 256),
            load_in_4bit  = cfg.get("load_in_4bit", True),
            dtype  = None,
        )
        FastLanguageModel.for_inference(self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, message: str) -> str:
        prompt = _INFER_TEMPLATE.format(message=message.strip())
        enc  = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens = 24,
                do_sample = False,
                use_cache  = True,
            )
        new_tok = out[0][enc["input_ids"].shape[1]:]
        raw     = self.tokenizer.decode(new_tok, skip_special_tokens=True).strip()
        predicted_label = raw.split("\n")[0].strip()
        return predicted_label


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/inference.yaml")
    p.add_argument("--message", required=True)
    args = p.parse_args()

    clf = IntentClassification(args.config)
    result = clf(args.message)
    print(f"Intent: {result}")
