#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

REL_PROMPT = {
    'system': """You are a knowledge graph relationship extraction agent. Your task is to extract structured relationships from simple sentences to create knowledge graph triples. Each triple must contain two entities and the relationship between them.""",
    'user': """Process:
- Analyze the sentence structure and identify key components.
- Extract all meaningful entities (nouns, noun phrases, proper nouns, concepts).
- Identify relationships between entities based on verbs, prepositions, and semantic meaning.
- Form triples (Entity 1 -> Relationship -> Entity 2).
- Validate that each triple captures meaningful semantic information.

Examples:

Input: "Regulating miR-497-5p provides a potential targeted therapy for lung cancer treatment."
Output:
[{
"Entity 1": "regulating miR-497-5p",
"Entity 2": "lung cancer targeted treatment",
"Relationship": "provides"
}]

Input: "The activation of caspase signal pathway was the reason for stronger apoptosis."
Output:
[{
"Entity 1": "activation of caspase signal pathway",
"Entity 2": "stronger apoptosis",
"Relationship": "was the reason for"
}]

Input: "With clinical significance features selection, over-sampling methods achieved the highest AUC results."
Output:
[{
"Entity 1": "clinical significance features selection",
"Entity 2": "over-sampling methods",
"Relationship": "with"
},
{
"Entity 1": "over-sampling methods",
"Entity 2": "highest AUC results",
"Relationship": "achieved"
}]

Now extract knowledge graph relationships from this sentence: {sentence}

Output only valid JSON (a list of objects with keys: "Entity 1", "Relationship", "Entity 2")."""
}

def build_prompt(tpl: Dict[str, str], sentence: str) -> str:
    return tpl["system"] + "\n\n" + tpl["user"].replace("{sentence}", sentence)

def extract_relationships(text: str):
    """
    Parse a JSON list of triples from model output.
    Accepts plain JSON or JSON wrapped in a code block. Returns (list, error_msg).
    """
    try:
        raw = text.strip()
        # strip code fences if present
        if raw.startswith("```"):
            parts = raw.splitlines()
            if len(parts) >= 3:
                raw = "\n".join(parts[1:-1]).strip()

        # First direct attempt
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj = [obj]
        if isinstance(obj, list):
            # normalize keys and filter invalids
            triples = []
            for it in obj:
                if not isinstance(it, dict):
                    continue
                e1 = it.get("Entity 1") or it.get("entity_1") or it.get("head") or it.get("subject")
                rel = it.get("Relationship") or it.get("relation") or it.get("predicate")
                e2 = it.get("Entity 2") or it.get("entity_2") or it.get("tail") or it.get("object")
                if e1 and rel and e2:
                    triples.append({"Entity 1": str(e1).strip(), "Relationship": str(rel).strip(), "Entity 2": str(e2).strip()})
            return triples, ""
        # fallback to bracket search
        raise ValueError("Not a JSON list/dict")
    except Exception:
        # bracket search fallback
        try:
            s = text.find('[')
            e = text.rfind(']')
            if s != -1 and e != -1 and e > s:
                obj = json.loads(text[s:e+1])
                if isinstance(obj, dict):
                    obj = [obj]
                triples = []
                for it in obj:
                    if not isinstance(it, dict):
                        continue
                    e1 = it.get("Entity 1") or it.get("entity_1") or it.get("head") or it.get("subject")
                    rel = it.get("Relationship") or it.get("relation") or it.get("predicate")
                    e2 = it.get("Entity 2") or it.get("entity_2") or it.get("tail") or it.get("object")
                    if e1 and rel and e2:
                        triples.append({"Entity 1": str(e1).strip(), "Relationship": str(rel).strip(), "Entity 2": str(e2).strip()})
                return triples, ""
            return [], "No JSON found"
        except Exception as ex:
            return [], f"Parse error: {ex}"

def load_llm(model_name: str, four_bit: bool = True):
    """
    Load a causal LM with accelerate (device_map='auto').
    Avoid passing a `device` into the pipeline (accelerate manages placement).
    """
    quant = None
    if four_bit:
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            quant = None

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", None) or getattr(tok, "sep_token", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant,
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok)  # Do not pass `device` here
    return pipe, tok

def main():
    ap = argparse.ArgumentParser(description="Extract KG triples from simple sentences (fallback to OriginalSentence if simple is empty).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out_path", required=True, help="Base output CSV path (will also create <out>.exploded.csv)")
    ap.add_argument("--id-cols", default="PMID,Title,csv_name,SentenceIndex,SimpleIndex", help="Comma-separated columns to carry over")
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="HF model id or local path")
    ap.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    ap.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    required = ["SimpleSentence", "OriginalSentence"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column '{c}'. Found columns: {list(df.columns)}")

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    keep_cols = [c for c in id_cols if c in df.columns]

    # Choose sentence: SimpleSentence if non-empty else OriginalSentence; if still empty, skip
    use_rows = []
    for i, row in df.iterrows():
        simple = str(row["SimpleSentence"]).strip() if pd.notna(row["SimpleSentence"]) else ""
        original = str(row["OriginalSentence"]).strip() if pd.notna(row["OriginalSentence"]) else ""
        chosen = simple if simple else (original if original else "")
        if not chosen:
            continue
        rec = {k: row[k] for k in keep_cols}
        rec["ChosenSentence"] = chosen
        use_rows.append(rec)

    if not use_rows:
        out_path = Path(args.out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=(keep_cols + ["ChosenSentence","Relationships","llm_raw_output","llm_error"])).to_csv(out_path, index=False)
        pd.DataFrame(columns=(keep_cols + ["ChosenSentence","Entity 1","Relationship","Entity 2"])).to_csv(out_path.with_suffix(out_path.suffix + ".exploded.csv"), index=False)
        print("No sentences to process. Wrote empty outputs.")
        return

    work_df = pd.DataFrame(use_rows, columns=(keep_cols + ["ChosenSentence"]))

    pipe, tok = load_llm(args.model, four_bit=True)

    prompts = [build_prompt(REL_PROMPT, s) for s in work_df["ChosenSentence"].tolist()]

    raw_texts = []
    triples_list = []
    errors = []

    print(f"Processing {len(prompts)} sentences...")
    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(prompts), bs), desc="Extracting"):
        batch_prompts = prompts[i:i+bs]
        gen = pipe(
            batch_prompts,
            max_new_tokens=args.max_new_tokens,
            truncation=True,
            return_full_text=False,
            pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
        )

        # Normalize pipeline outputs
        outs = []
        for item in gen:
            if isinstance(item, list) and item and isinstance(item[0], dict) and "generated_text" in item[0]:
                outs.append(item[0]["generated_text"])
            elif isinstance(item, dict) and "generated_text" in item:
                outs.append(item["generated_text"])
            else:
                outs.append("")

        for txt in outs:
            triples, err = extract_relationships(txt)
            raw_texts.append(txt)
            triples_list.append(triples)
            errors.append(err)

    out_df = work_df.copy()
    out_df["Relationships"] = [json.dumps(x, ensure_ascii=False) for x in triples_list]
    out_df["llm_raw_output"] = raw_texts
    out_df["llm_error"] = errors

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Exploded version: 1 row per triple
    rows = []
    for idx, triples in enumerate(triples_list):
        if not triples:
            continue
        meta = {k: out_df.iloc[idx][k] for k in keep_cols}
        sent = out_df.iloc[idx]["ChosenSentence"]
        for t in triples:
            r = dict(meta)
            r["ChosenSentence"] = sent
            r["Entity 1"] = t.get("Entity 1", "")
            r["Relationship"] = t.get("Relationship", "")
            r["Entity 2"] = t.get("Entity 2", "")
            rows.append(r)

    exploded_df = pd.DataFrame(rows, columns=(keep_cols + ["ChosenSentence","Entity 1","Relationship","Entity 2"]))
    exploded_out = out_path.with_suffix(out_path.suffix + ".exploded.csv")
    exploded_df.to_csv(exploded_out, index=False)

    print(f"Wrote {len(out_df)} sentences to: {out_path}")
    print(f"Wrote {len(exploded_df)} triples to: {exploded_out}")

if __name__ == "__main__":
    main()
