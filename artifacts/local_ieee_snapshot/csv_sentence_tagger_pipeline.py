#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

LABEL_MAP = {
    0: "Complex Sentence",
    1: "Compound Sentence",
    2: "Compound-Complex Sentence",
    3: "Incomplete Sentence",
    4: "Simple Sentence",
}
LABELS = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]

def setup_nltk_quiet():
    try:
        sent_tokenize("Test sentence.")
        return
    except LookupError:
        pass
    for res in ["punkt", "punkt_tab"]:
        try:
            nltk.download(res, quiet=True)
            sent_tokenize("Test sentence.")
            return
        except Exception:
            continue
    # will raise if still missing
    sent_tokenize("Test sentence.")

def separate_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = sent_tokenize(text.strip())
    return [s.strip() for s in sents if s.strip()]

def load_top5(json_path: str) -> List[str]:
    """
    Return up to 5 model paths, resolved as LOCAL directories relative to the JSON file
    (or absolute if already absolute). If a resolved local path doesn't exist, keep the
    raw string so we may try it as an HF repo id.
    """
    jp = Path(json_path)
    base = jp.parent
    with open(jp, "r") as f:
        arr = json.load(f)

    def _pick_str(item):
        if isinstance(item, list):
            for x in item:
                if isinstance(x, str):
                    return x
            return None
        return item if isinstance(item, str) else None

    resolved = []
    for item in arr[:5]:
        s = _pick_str(item)
        if not s:
            continue
        p = Path(s)
        # absolute local path
        if p.is_absolute() and p.exists():
            resolved.append(str(p.resolve()))
            continue
        # relative to JSON dir
        p1 = base / p
        if p1.exists():
            resolved.append(str(p1.resolve()))
            continue
        # one level up from JSON dir (common layout)
        p2 = base.parent / p
        if p2.exists():
            resolved.append(str(p2.resolve()))
            continue
        # fallback: keep raw string (may be an HF repo id)
        resolved.append(s)
    return resolved

def tag_sentences(sentences, model_paths, device_index=0, batch_size=32, max_len=128):
    use_gpu = (device_index >= 0) and torch.cuda.is_available()
    device = torch.device(f"cuda:{device_index}" if use_gpu else "cpu")
    if not model_paths:
        raise SystemExit("No model paths provided for tagging.")

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    print("Resolved model paths (local preferred):")
    for mp in model_paths:
        print(" -", mp, "| local:", Path(mp).exists())

    models = []
    for mpath in model_paths:
        try:
            if Path(mpath).exists():
                model = AutoModelForSequenceClassification.from_pretrained(mpath, local_files_only=True)
            else:
                # treat as HF repo id only if not a local path
                model = AutoModelForSequenceClassification.from_pretrained(mpath)
        except Exception as e:
            raise SystemExit(f"Failed to load model '{mpath}': {e}")

        model.config.id2label = {i: LABEL_MAP[i] for i in range(len(LABEL_MAP))}
        model.config.label2id = {v: k for k, v in LABEL_MAP.items()}
        model.to(device).eval()
        models.append(model)

    preds = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), bs), desc="Tagging"):
            batch_texts = sentences[i:i+bs]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            probs_sum = None
            for model in models:
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else (probs_sum + probs)
            avg_probs = (probs_sum / len(models)).detach().cpu().numpy()
            batch_idx = np.argmax(avg_probs, axis=1).tolist()
            batch_labels = [LABELS[j] for j in batch_idx]
            preds.extend(batch_labels)
    return preds

def main():
    ap = argparse.ArgumentParser(description="Split abstracts into sentences, explode, and tag them.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path (e.g., Out_coref.csv).")
    ap.add_argument("--text-col", default="Abstract", help="Column name containing the finished abstract text.")
    ap.add_argument("--id-cols", default="", help="Comma-separated list of metadata columns (e.g., PMID,Title,csv_name).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path for the sentence-tagged dataframe.")
    ap.add_argument("--no-tag", action="store_true", help="Only split into sentences (skip tagging).")
    ap.add_argument("--json-path", default="/data0/projects/Causal_and_Agentic_AI/knowledgegraph2025/sentencetagger/bertlarge/bertlarge_output/top_models_bertlarge.json",
                    help="Path to JSON listing top-5 model folders or repo ids.")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (-1 for CPU).")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for tagging.")
    ap.add_argument("--max-len", type=int, default=128, help="Max sequence length for tagging.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {args.in_path}.")

    setup_nltk_quiet()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    keep_cols = [c for c in id_cols if c in df.columns]

    sent_rows = []
    for _, row in df.iterrows():
        text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""
        sents = separate_sentences(text)
        for s_idx, s in enumerate(sents):
            r = {"Sentence": s, "SentenceIndex": s_idx}
            for c in keep_cols:
                r[c] = row[c]
            sent_rows.append(r)

    out_df = pd.DataFrame(sent_rows, columns=(keep_cols + ["SentenceIndex", "Sentence"]))
    if args.no_tag:
        out_df.to_csv(args.out_path, index=False)
        print(f"Wrote sentences only -> {args.out_path}")
        return

    model_paths = load_top5(args.json_path)
    preds = tag_sentences(
        sentences=out_df["Sentence"].tolist(),
        model_paths=model_paths,
        device_index=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    out_df["tagged"] = preds
    out_df.to_csv(args.out_path, index=False)
    print(f"Saved tagged CSV to: {args.out_path}")

if __name__ == "__main__":
    main()
