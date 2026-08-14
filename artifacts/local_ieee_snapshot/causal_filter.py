#!/usr/bin/env python3

import argparse
import pandas as pd
import re
import sys
import unicodedata
from pathlib import Path

STRICT_MARKERS = [
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bas a result\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bleads? to\b",
    r"\bled to\b",
    r"\bresult(?:s|ed)? in\b",
    r"\bcaus(?:e|es|ed|al)\b",
]

BROAD_MARKERS = STRICT_MARKERS + [
    r"\bcontribut(?:e|es|ed|ing) to\b",
    r"\bpredict(?:s|ed|ive)?\b",
    r"\binfluenc(?:e|es|ed|ing)\b",
    r"\bimpact(?:s|ed|ing)?\b",
    r"\bassociated with\b",
    r"\bincrease(?:s|d|ing)? the risk\b",
    r"\breduc(?:e|es|ed|ing)? the risk\b",
]

def build_regex(markers):
    pattern = re.compile("(" + "|".join(markers) + ")", flags=re.IGNORECASE)
    return pattern

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("Œµ", "ε")  # fix garbled epsilon (e.g., APOE-ε4)
    t = t.replace("\u00ad", "")  # soft hyphen
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    t = re.sub(r"([,;:.!?])\1+", r"\1", t)
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.IGNORECASE)
    return t

def find_first_marker(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(text)
    if not m:
        return None
    return m.group(0).lower()

def main():
    parser = argparse.ArgumentParser(
        description="Filter a CSV by causal markers in a text column, clean the text, and keep ALL original columns."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV (default: <input>_causal.csv)")
    parser.add_argument("-c", "--column", default="SimpleSentence", help="Name of the text column (default: SimpleSentence)")
    parser.add_argument("--mode", choices=["strict", "broad"], default="broad", help="Causal marker sensitivity (default: broad)")
    parser.add_argument("--extra-markers", default="", help="Comma-separated extra markers to include (e.g., 'mediates,accounts for')")
    parser.add_argument("--min-chars", type=int, default=0, help="Drop rows where cleaned text is shorter than this many characters")
    parser.add_argument("--na-action", choices=["drop", "keep"], default="drop", help="What to do with NA texts before filtering (default: drop)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_causal.csv")

    # Choose marker set
    markers = STRICT_MARKERS if args.mode == "strict" else BROAD_MARKERS
    if args.extra_markers.strip():
        extras = [m.strip() for m in args.extra_markers.split(",") if m.strip()]
        for m in extras:
            if m.startswith("\\") or "(" in m or "|" in m:
                markers.append(m)  # assume raw regex
            else:
                token = re.escape(m)
                if " " in m:
                    markers.append(rf"\b{token}\b")
                else:
                    markers.append(rf"\b{token}\b")

    pattern = build_regex(markers)

    # Load CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.column not in df.columns:
        print(f"ERROR: Column '{args.column}' not found in CSV. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Optionally drop NA before processing
    if args.na_action == "drop":
        df = df.dropna(subset=[args.column])

    # Clean text
    cleaned = df[args.column].map(normalize_text)

    # Drop too-short rows if requested
    if args.min_chars and args.min_chars > 0:
        mask_len = cleaned.str.len() >= args.min_chars
        cleaned = cleaned[mask_len]
        df = df.loc[mask_len]

    # Find causal markers
    matched = cleaned.map(lambda t: find_first_marker(t, pattern))

    # Keep only rows where a marker was found
    keep_mask = matched.notna()
    filtered = df.loc[keep_mask].copy()

    # Attach helper columns while preserving ALL original columns
    filtered[f"{args.column}_clean"] = cleaned.loc[keep_mask].values
    filtered["causal_marker"] = matched.loc[keep_mask].values

    # Write CSV
    try:
        filtered.to_csv(out_path, index=False)
    except Exception as e:
        print(f"ERROR: Failed to write CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(filtered)} rows to {out_path} (kept all original columns, plus '{args.column}_clean' and 'causal_marker').")

if __name__ == "__main__":
    main()
