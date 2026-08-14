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

# ---------------------------
# Human-readable labels
# ---------------------------
LABEL_MAP = {
    0: "Complex Sentence",
    1: "Compound Sentence",
    2: "Compound-Complex Sentence",
    3: "Incomplete Sentence",
    4: "Simple Sentence",
}

# ---------------------------
# NLTK setup
# ---------------------------
def setup_nltk_quiet():
    """Ensure sentence tokenizer resources are available (punkt/punkt_tab)."""
    try:
        sent_tokenize("Test sentence.")
        return
    except LookupError:
        pass

    for res in ("punkt", "punkt_tab"):
        try:
            nltk.download(res, quiet=True)
            sent_tokenize("Test sentence.")
            return
        except Exception:
            continue
    sent_tokenize("Test sentence.")  # will raise if still missing

def separate_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = sent_tokenize(text.strip())
    return [s.strip() for s in sents if s.strip()]

# ---------------------------
# Model path resolution
# ---------------------------
def load_top5(json_path: str) -> List[str]:
    """
    Return up to 5 model dirs. Resolve relative to JSON location; if not found, keep as-is.
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

        # Absolute path present?
        if p.is_absolute() and p.exists():
            resolved.append(str(p.resolve()))
            continue

        # Try relative to JSON dir
        p1 = base / p
        if p1.exists():
            resolved.append(str(p1.resolve()))
            continue

        # Try one level up (common layout)
        p2 = base.parent / p
        if p2.exists():
            resolved.append(str(p2.resolve()))
            continue

        # Keep raw (could be HF repo id, etc.)
        resolved.append(s)

    return resolved

def maybe_checkpoint_dir(model_dir: Path) -> Path:
    """
    If the fold dir contains a single 'checkpoint-xxxx' subdir (sometimes used during saving),
    prefer that as the load path; otherwise return the original dir.
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return model_dir
    candidates = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    return candidates[0] if len(candidates) == 1 else model_dir

# ---------------------------
# Inference (shared tokenizer)
# ---------------------------
def tag_sentences(
    sentences: List[str],
    model_paths: List[str],
    device_index: int = 0,
    batch_size: int = 32,
    max_len: int = 128,
) -> List[str]:
    """
    Predict labels for a list of sentences using probability-averaged ensemble across model_paths.
    Uses a single, known-good tokenizer ('bert-large-uncased') to avoid missing tokenizer files in fold dirs.
    """
    if not model_paths:
        raise SystemExit("No model paths provided for tagging.")

    use_gpu = (device_index >= 0) and torch.cuda.is_available()
    device = torch.device(f"cuda:{device_index}" if use_gpu else "cpu")

    # Use a stable tokenizer (pretrained base) across all folds
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token or tokenizer.eos_token or "[PAD]"

    # Be conservative across folds
    max_len = int(min(256, max_len))

    # Load models (weights only) from each fold
    models = []
    for mpath in model_paths:
        try:
            load_path = Path(mpath)
            if load_path.exists():
                load_path = maybe_checkpoint_dir(load_path)
                model = AutoModelForSequenceClassification.from_pretrained(str(load_path), local_files_only=True)
            else:
                # As a last resort, try remote id/path
                model = AutoModelForSequenceClassification.from_pretrained(mpath)
        except Exception as e:
            raise SystemExit(f"Failed to load model '{mpath}': {e}")

        # Force our human-readable labels
        model.config.id2label = {i: LABEL_MAP[i] for i in range(len(LABEL_MAP))}
        model.config.label2id = {v: k for k, v in LABEL_MAP.items()}

        # Align pad id if needed
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        model.to(device).eval()
        models.append(model)

    preds = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), bs), desc="Tagging"):
            batch_texts = sentences[i:i + bs]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len,
            )
            # Provide token_type_ids if missing and model expects them
            if "token_type_ids" not in enc or enc["token_type_ids"] is None:
                # BERT expects token_type_ids; if tokenizer omitted them, add zeros
                enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

            enc = {k: v.to(device) for k, v in enc.items()}

            probs_sum = None
            for model in models:
                logits = model(**enc).logits  # [B, C]
                probs = torch.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            avg_probs = (probs_sum / len(models)).detach().cpu().numpy()  # [B, C]
            batch_idx = np.argmax(avg_probs, axis=1).tolist()
            preds.extend([LABEL_MAP[j] for j in batch_idx])

    return preds

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Split abstracts into sentences with NLTK, explode, and tag them with an ensemble of top-5 folds."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path.")
    ap.add_argument("--text-col", default="Abstract", help="Column containing the full text to be sentence-split.")
    ap.add_argument("--id-cols", default="", help="Comma-separated columns to carry through (e.g., PMID,Title).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path.")
    ap.add_argument("--no-tag", action="store_true", help="Only split into sentences (skip tagging).")
    ap.add_argument("--json-path", required=True, help="Path to top_models_bertlarge.json.")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (-1 for CPU).")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for tagging.")
    ap.add_argument("--max-len", type=int, default=128, help="Max sequence length (capped to 256).")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {args.in_path}.")

    setup_nltk_quiet()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    keep_cols = [c for c in id_cols if c in df.columns]

    # Explode to sentence level
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
    if not model_paths:
        raise SystemExit(f"No model paths found in JSON: {args.json_path}")

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
