#!/usr/bin/env python3
import argparse
import re
import sys
import pandas as pd

def tokenize(text, strip_punct=True):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if strip_punct:
        return [re.sub(r'^\W+|\W+$', '', token)
                for token in text.split()
                if re.sub(r'^\W+|\W+$', '', token)]
    return text.split()

def tokens_with_indices(text, keep_punct=True):
    toks = tokenize(text, strip_punct=not keep_punct)
    return [(t, i) for i, t in enumerate(toks)]

def format_pairs(pairs):
    return ", ".join(f'("{t}", {i})' for (t, i) in pairs)

def transform_abstract(text):
    pairs = tokens_with_indices(text, keep_punct=True)
    return format_pairs(pairs)

def main():
    p = argparse.ArgumentParser(description="Rename Abstract->OldAbstract and write a new Abstract column (token-index pairs).")
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    p.add_argument("--rename-only", action="store_true",
                   help="Only rename Abstract->OldAbstract and copy values to Abstract unchanged")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.in_path, encoding=args.encoding, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"ERROR: failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if "Abstract" not in df.columns:
        print("ERROR: Column 'Abstract' not found in the input CSV.", file=sys.stderr)
        sys.exit(2)

    df["OldAbstract"] = df["Abstract"]

    if args.rename_only:
        df["Abstract"] = df["OldAbstract"]
    else:
        df["Abstract"] = df["OldAbstract"].astype(str).apply(transform_abstract)

    try:
        df.to_csv(args.out_path, index=False, encoding=args.encoding)
    except Exception as e:
        print(f"ERROR: failed to write CSV: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote {len(df)} rows -> {args.out_path}")

if __name__ == "__main__":
    main()
