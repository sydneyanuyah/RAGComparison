#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Six-mode MCQ RAG evaluation on a LOCAL server with a Hugging Face causal LM.

Modes
1) No-RAG
2) G1 (g1.csv)
3) G2 (g2.csv)
4) G1+G2
5) G3 (g3.csv)
6) G1+G2+G3

Inputs
- questions_final_schema_balanced.json  (dynamic 4/5 choices; optional numbered options)
- g1.csv, g2.csv, g3.csv                (columns: E1, Relation, E2)

Outputs (in --outdir)
- predictions_six_modes_local.csv
- metrics_six_modes_local.(csv|json)

Example
python rag_eval_local.py   --model_name mistralai/Mistral-7B-Instruct-v0.2   --dataset questions_final_schema_balanced.json   --g1 g1.csv --g2 g2.csv --g3 g3.csv   --outdir rag_local_mistral --temperature 0.3 --max_new_tokens 16 --top_k 6
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import joblib
from scipy import sparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)

REL_PROMPT = {
    "system": (
        "You are answering a multiple-choice question. "
        "Return ONLY one uppercase letter from this set: {allowed}. "
        "Do not include explanations or extra text."
    ),
    "user_no_rag": (
        "{system}\n\n"
        "{question_block}\n\n"
        "Answer:"
    ),
    "user_rag": (
        "Use ONLY the provided context to answer. If the answer is not implied, guess your best.\n"
        "{system}\n\n"
        "Context:\n{context_block}\n\n"
        "{question_block}\n\n"
        "Answer:"
    ),
}

def load_llm(model_name: str, four_bit: bool = True):
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
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok)
    return pipe, tok

def load_mcq_json(path: str) -> pd.DataFrame:
    def get_answer_letter(item):
        if isinstance(item.get("answer"), dict) and "letter" in item["answer"]:
            return item["answer"]["letter"]
        return item.get("answer_letter")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        qid = item.get("id")
        qtext = (item.get("question") or item.get("prompt") or "").strip()
        options = item.get("options", {}) or {}
        choices = item.get("choices", {}) or {}
        ans_letter = get_answer_letter(item)
        rows.append({
            "id": qid,
            "question": qtext,
            "options": options,
            "choices": choices,
            "answer_letter": ans_letter
        })

    df = pd.DataFrame(rows)
    try:
        df["id_int"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.sort_values(["id_int","id"]).drop(columns=["id_int"]).reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)
    return df

def load_facts_from_csv(path: str):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    for col in ["E1", "Relation", "E2"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    def _fix(s):
        return str(s).replace("Œ≤", "β").strip()
    df["E1"] = df["E1"].map(_fix)
    df["Relation"] = df["Relation"].map(_fix)
    df["E2"] = df["E2"].map(_fix)
    df["fact"] = df["E1"] + " " + df["Relation"] + " " + df["E2"] + "."
    return df, df["fact"].tolist()

def build_tfidf(corpus: List[str]):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    mat = vec.fit_transform(corpus)
    return vec, mat

def retrieve(query: str, vec, mat, facts: List[str], k: int = 6) -> List[Dict[str, Any]]:
    from sklearn.metrics.pairwise import cosine_similarity
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).ravel()
    idx = np.argsort(-sims)[:k]
    return [{"row": int(i), "score": float(sims[i]), "text": facts[i]} for i in idx]

def letters_for(item: Dict[str, Any]) -> List[str]:
    letters = [k for k in (item.get("choices") or {}).keys() if isinstance(k, str)]
    return sorted(letters, key=lambda x: x)

def mcq_block(item: Dict[str, Any]) -> str:
    q = (item.get("question") or "").strip()
    opts = item.get("options", {}) or {}
    choices = item.get("choices", {}) or {}
    lines = [f"Question: {q}", ""]
    if len(opts) > 0:
        lines.append("Options (numbered):")
        def key_sort(x):
            try:
                return int(str(x))
            except:
                return str(x)
        for k in sorted(opts, key=key_sort):
            lines.append(f"{k}. {opts[k]}")
        lines.append("")
    lines.append("Choices:")
    for k in letters_for(item):
        lines.append(f"{k}: {choices[k]}")
    return "\n".join(lines)

def build_no_rag_prompt(item: Dict[str, Any]) -> str:
    allowed = letters_for(item)
    allowed_str = ", ".join(allowed)
    sys = REL_PROMPT["system"].format(allowed=allowed_str)
    qblk = mcq_block(item)
    return REL_PROMPT["user_no_rag"].format(system=sys, question_block=qblk)

def build_rag_prompt(item: Dict[str, Any], ctx_hits: List[Dict[str, Any]], source_label: str) -> str:
    allowed = letters_for(item)
    allowed_str = ", ".join(allowed)
    sys = REL_PROMPT["system"].format(allowed=allowed_str)
    ctx = "\n".join([f"- ({h['row']}) {h['text']}" for h in ctx_hits]) or "(no relevant context)"
    qblk = mcq_block(item)
    return REL_PROMPT["user_rag"].format(system=sys, context_block=ctx, question_block=qblk)

def parse_letter(text: str, allowed_letters: List[str]) -> Optional[str]:
    if not text:
        return None
    import re
    pat = re.compile(r"\b([" + "".join(allowed_letters) + r"])\b")
    s = text.strip().upper()
    m = None
    for m in pat.finditer(s):
        pass
    if m:
        return m.group(1)
    for ch in s:
        if ch in set(allowed_letters):
            return ch
    return None

def generate_one(pipe: TextGenerationPipeline, prompt: str, temperature: float, max_new_tokens: int) -> str:
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(0.0, float(temperature)),
        top_p=0.9,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
        num_return_sequences=1,
    )
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
        return out[0]["generated_text"][len(prompt):]
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], str):
        return out[0]
    return str(out)

def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
    }

def main():
    ap = argparse.ArgumentParser(description="Local six-mode MCQ RAG evaluation")
    ap.add_argument("--model_name", required=True, type=str, help="HF model name (e.g., mistralai/Mistral-7B-Instruct-v0.2)")
    ap.add_argument("--dataset", default="questions_final_schema_balanced.json", type=str)
    ap.add_argument("--g1", required=True, type=str, help="Path to g1.csv")
    ap.add_argument("--g2", required=True, type=str, help="Path to g2.csv")
    ap.add_argument("--g3", required=True, type=str, help="Path to g3.csv")
    ap.add_argument("--outdir", default="rag_local_outputs", type=str)
    ap.add_argument("--temperature", default=0.2, type=float)
    ap.add_argument("--max_new_tokens", default=16, type=int)
    ap.add_argument("--top_k", default=6, type=int, help="Retrieval K")
    ap.add_argument("--four_bit", action="store_true", help="Load in 4-bit quantization (if available)")
    ap.add_argument("--limit", default=0, type=int, help="Limit number of questions for a quick run (0 = all)")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    np.random.seed(args.seed)
    import random as pyrand
    pyrand.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model_name}")
    pipe, tok = load_llm(args.model_name, four_bit=args.four_bit)

    print(f"Loading dataset: {args.dataset}")
    df_mcq = load_mcq_json(args.dataset)
    if args.limit and args.limit > 0:
        df_mcq = df_mcq.head(args.limit).copy()

    print("Loading corpora and building TF-IDF indexes...")
    df_g1, facts_g1 = load_facts_from_csv(args.g1)
    df_g2, facts_g2 = load_facts_from_csv(args.g2)
    df_g3, facts_g3 = load_facts_from_csv(args.g3)

    facts_g12  = facts_g1 + facts_g2
    facts_g123 = facts_g1 + facts_g2 + facts_g3

    vec_g1,   mat_g1   = build_tfidf(facts_g1)
    vec_g2,   mat_g2   = build_tfidf(facts_g2)
    vec_g3,   mat_g3   = build_tfidf(facts_g3)
    vec_g12,  mat_g12  = build_tfidf(facts_g12)
    vec_g123, mat_g123 = build_tfidf(facts_g123)

    joblib.dump(vec_g1,   os.path.join(args.outdir, "tfidf_g1.joblib"))
    joblib.dump(vec_g2,   os.path.join(args.outdir, "tfidf_g2.joblib"))
    joblib.dump(vec_g3,   os.path.join(args.outdir, "tfidf_g3.joblib"))
    joblib.dump(vec_g12,  os.path.join(args.outdir, "tfidf_g12.joblib"))
    joblib.dump(vec_g123, os.path.join(args.outdir, "tfidf_g123.joblib"))
    sparse.save_npz(os.path.join(args.outdir, "mat_g1.npz"),   mat_g1)
    sparse.save_npz(os.path.join(args.outdir, "mat_g2.npz"),   mat_g2)
    sparse.save_npz(os.path.join(args.outdir, "mat_g3.npz"),   mat_g3)
    sparse.save_npz(os.path.join(args.outdir, "mat_g12.npz"),  mat_g12)
    sparse.save_npz(os.path.join(args.outdir, "mat_g123.npz"), mat_g123)

    print("Running evaluation...")
    results: List[Dict[str, Any]] = []
    for _, item in tqdm(df_mcq.iterrows(), total=len(df_mcq), ncols=100):
        allowed = letters_for(item)

        out_nr = generate_one(pipe, build_no_rag_prompt(item), args.temperature, args.max_new_tokens)
        pred_nr = parse_letter(out_nr, allowed) or ""

        hits_g1 = retrieve(item["question"], vec_g1, mat_g1, facts_g1, k=args.top_k)
        out_g1  = generate_one(pipe, build_rag_prompt(item, hits_g1, "G1"), args.temperature, args.max_new_tokens)
        pred_g1 = parse_letter(out_g1, allowed) or ""

        hits_g2 = retrieve(item["question"], vec_g2, mat_g2, facts_g2, k=args.top_k)
        out_g2  = generate_one(pipe, build_rag_prompt(item, hits_g2, "G2"), args.temperature, args.max_new_tokens)
        pred_g2 = parse_letter(out_g2, allowed) or ""

        hits_g12 = retrieve(item["question"], vec_g12, mat_g12, facts_g12, k=args.top_k)
        out_g12  = generate_one(pipe, build_rag_prompt(item, hits_g12, "G1+G2"), args.temperature, args.max_new_tokens)
        pred_g12 = parse_letter(out_g12, allowed) or ""

        hits_g3 = retrieve(item["question"], vec_g3, mat_g3, facts_g3, k=args.top_k)
        out_g3  = generate_one(pipe, build_rag_prompt(item, hits_g3, "G3"), args.temperature, args.max_new_tokens)
        pred_g3 = parse_letter(out_g3, allowed) or ""

        hits_g123 = retrieve(item["question"], vec_g123, mat_g123, facts_g123, k=args.top_k)
        out_g123  = generate_one(pipe, build_rag_prompt(item, hits_g123, "G1+G2+G3"), args.temperature, args.max_new_tokens)
        pred_g123 = parse_letter(out_g123, allowed) or ""

        gold = item["answer_letter"]
        rid  = item["id"]
        try:
            rid = int(rid)
        except Exception:
            pass

        results.append({
            "id": rid,
            "gold": gold,
            "pred_no_rag": pred_nr,
            "pred_g1": pred_g1,
            "pred_g2": pred_g2,
            "pred_g12": pred_g12,
            "pred_g3": pred_g3,
            "pred_g123": pred_g123,
            "correct_no_rag": int(pred_nr == gold),
            "correct_g1": int(pred_g1 == gold),
            "correct_g2": int(pred_g2 == gold),
            "correct_g12": int(pred_g12 == gold),
            "correct_g3": int(pred_g3 == gold),
            "correct_g123": int(pred_g123 == gold),
            "raw_no_rag": out_nr,
            "raw_g1": out_g1,
            "raw_g2": out_g2,
            "raw_g12": out_g12,
            "raw_g3": out_g3,
            "raw_g123": out_g123,
        })

    df_res = pd.DataFrame(results)
    try:
        df_res = df_res.sort_values("id").reset_index(drop=True)
    except Exception:
        pass

    y_true = df_res["gold"].tolist()
    modes  = {
        "No-RAG": df_res["pred_no_rag"].tolist(),
        "G1":     df_res["pred_g1"].tolist(),
        "G2":     df_res["pred_g2"].tolist(),
        "G1+G2":  df_res["pred_g12"].tolist(),
        "G3":     df_res["pred_g3"].tolist(),
        "G1+G2+G3": df_res["pred_g123"].tolist(),
    }
    summary_rows = [compute_metrics(y_true, preds) for preds in modes.values()]
    summary = pd.DataFrame(summary_rows, index=list(modes.keys()))

    preds_path = os.path.join(args.outdir, "predictions_six_modes_local.csv")
    metrics_csv = os.path.join(args.outdir, "metrics_six_modes_local.csv")
    metrics_json = os.path.join(args.outdir, "metrics_six_modes_local.json")

    cols = [
        "id","gold","pred_no_rag","pred_g1","pred_g2","pred_g12","pred_g3","pred_g123",
        "correct_no_rag","correct_g1","correct_g2","correct_g12","correct_g3","correct_g123"
    ]
    df_res[cols].to_csv(preds_path, index=False)
    summary.to_csv(metrics_csv)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.to_dict(orient="index").items()}, f, indent=2)

    print(f"\nSaved:\n- {preds_path}\n- {metrics_csv}\n- {metrics_json}")

if __name__ == "__main__":
    main()
