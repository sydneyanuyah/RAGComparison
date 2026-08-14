#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

CLASS_PREFIX = {
    "Complex Sentence": "complex_",
    "Compound Sentence": "compound_",
    "Compound-Complex Sentence": "compound_complex_",
    "Incomplete Sentence": "incomplete_",
    "Simple Sentence": "simple_",
}

def main():
    ap = argparse.ArgumentParser(
        description="Copy labeled CSV into sentence_tagged/ with class-based prefixes for all present classes."
    )
    ap.add_argument("--csv", required=True, help="Path to labeled CSV (has 'tagged' or 'predicted_label').")
    ap.add_argument("--subdir", default="sentence_tagged", help="Subfolder to write into (default: sentence_tagged).")
    args = ap.parse_args()

    in_path = Path(args.csv).resolve()
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    # Detect label column
    label_col = None
    for c in ("tagged", "predicted_label"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise SystemExit("No 'tagged' or 'predicted_label' column found in the CSV.")

    # Prepare output dir
    out_dir = in_path.parent / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    basename = in_path.name
    present_classes = set(df[label_col].dropna().astype(str))

    saved_any = False
    for cls_name, prefix in CLASS_PREFIX.items():
        if cls_name in present_classes:
            out_path = out_dir / f"{prefix}{basename}"
            df.to_csv(out_path, index=False)
            print(f"Saved (class present: {cls_name}) -> {out_path}")
            saved_any = True

    if not saved_any:
        # No known classes found; just copy without prefix as a fallback
        fallback_path = out_dir / basename
        df.to_csv(fallback_path, index=False)
        print(f"No known classes found. Saved fallback -> {fallback_path}")

if __name__ == "__main__":
    main()
