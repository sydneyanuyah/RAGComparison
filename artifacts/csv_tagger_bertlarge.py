#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ---- Fixed human-readable label map ----
LABEL_MAP = {
    0: "Complex Sentence",
    1: "Compound Sentence",
    2: "Compound-Complex Sentence",
    3: "Incomplete Sentence",
    4: "Simple Sentence",
}
LABELS = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]

def load_top5(json_path: str):
    with open(json_path, "r") as f:
        arr = json.load(f)
    return [m[1] for m in arr[:5]]

def main():
    p = argparse.ArgumentParser(description="Batch classify sentences with top-5 BERT-large folds and write 'tagged' column.")
    p.add_argument("--csv_path", required=True, help="Input CSV file path.")
    p.add_argument("--text_column", required=True, help="Name of the column containing sentences.")
    p.add_argument("--out_path", default=None, help="Output CSV path (default: <csv_basename>_tagged.csv).")
    p.add_argument("--json_path", default="../knowledgegraph2025/sentencetagger/bertlarge/bertlarge_output/top_models_bertlarge.json",
                   help="Path to top_models_bertlarge.json")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (use -1 for CPU).")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    p.add_argument("--max_len", type=int, default=128, help="Max sequence length.")
    args = p.parse_args()

    # Resolve device
    use_gpu = (args.device >= 0) and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if use_gpu else "cpu")

    # I/O
    in_path = Path(args.csv_path)
    if args.out_path is None:
        out_path = in_path.with_name(in_path.stem + "_tagged.csv")
    else:
        out_path = Path(args.out_path)

    # Read data
    df = pd.read_csv(in_path)
    if args.text_column not in df.columns:
        raise SystemExit(f"Column '{args.text_column}' not found in CSV.")
    texts = df[args.text_column].astype(str).fillna("").tolist()

    # Load models and tokenizer
    model_paths = load_top5(args.json_path)
    if not model_paths:
        raise SystemExit("No models found in JSON.")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    models = []
    for mpath in model_paths:
        model = AutoModelForSequenceClassification.from_pretrained(mpath)
        # Make sure label maps match our desired names
        model.config.id2label = {i: LABEL_MAP[i] for i in range(len(LABEL_MAP))}
        model.config.label2id = {v: k for k, v in LABEL_MAP.items()}
        model.to(device).eval()
        models.append(model)

    # Inference (probability averaging across models)
    preds = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc="Predicting"):
            batch_texts = texts[i:i+bs]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # accumulate probabilities from each model
            probs_sum = None
            for model in models:
                logits = model(**enc).logits  # [B, C]
                probs = torch.softmax(logits, dim=-1)  # [B, C]
                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            avg_probs = (probs_sum / len(models)).detach().cpu().numpy()  # [B, C]
            batch_idx = np.argmax(avg_probs, axis=1).tolist()
            batch_labels = [LABELS[j] for j in batch_idx]
            preds.extend(batch_labels)

    # Write results
    df["tagged"] = preds
    df.to_csv(out_path, index=False)
    print(f"Saved tagged CSV to: {out_path}")

if __name__ == "__main__":
    main()
