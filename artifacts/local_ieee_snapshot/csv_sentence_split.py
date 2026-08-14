#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

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
    raise SystemExit("NLTK sentence tokenizer data not found and could not be downloaded. "
                     "If this machine is offline, run once: python -c \"import nltk; nltk.download('punkt')\"")

def separate_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = sent_tokenize(text.strip())
    return [s.strip() for s in sents if s and s.strip()]

def main():
    ap = argparse.ArgumentParser(description="Split CSV abstracts into sentences and EXPLODE rows (NLTK only).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path (e.g., Out_coref.csv).")
    ap.add_argument("--text-col", default="Abstract", help="Column name with text to split (default: Abstract).")
    ap.add_argument("--id-cols", default="", help="Comma-separated metadata columns to carry through (e.g., PMID,Title,csv_name).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path (e.g., Out_sentences.csv).")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {args.in_path}. Available: {list(df.columns)}")

    setup_nltk_quiet()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    keep_cols = [c for c in id_cols if c in df.columns]

    rows = []
    for _, row in df.iterrows():
        text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""
        sents = separate_sentences(text)
        for s_idx, s in enumerate(sents):
            out = {"SentenceIndex": s_idx, "Sentence": s}
            for c in keep_cols:
                out[c] = row[c]
            rows.append(out)

    out_df = pd.DataFrame(rows, columns=(keep_cols + ["SentenceIndex", "Sentence"]))
    out_df.to_csv(args.out_path, index=False)
    print(f"Wrote {len(out_df)} sentences to {args.out_path}")

if __name__ == "__main__":
    main()
