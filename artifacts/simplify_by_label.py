#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplify sentences based on a predicted_label column:
- Return 'Simple Sentence' and 'Incomplete Sentence' as-is (no LLM call).
- Simplify the rest using class-specific prompts:
  - 'Complex Sentence' -> complex prompt
  - 'Compound Sentence' -> compound prompt
  - 'Compound-Complex Sentence' -> compound-complex prompt
"""
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# ---------------- Prompts ----------------
PROMPT_COMPLEX = {
    'system': "You are an expert in sentence simplification. Your task is to convert complex sentences into simple sentences.",
    'user': (
        "Below is a step-by-step process. For each example, think step by step, then output only the simplified sentences "
        "in the form:\n"
        "S1 -> ... S2 -> ... ...\n"
        "one per line, and nothing else.\n\n"
        "Example 1:\n"
        "Input:\n"
        "\"A prospective cohort study was conducted in Leeds, UK, based on routinely collected data from a service that allowed patients with symptoms of lung cancer to request CXR.\"\n\n"
        "Chain-of-Thought:\n"
        "1. Identify the independent clause: \"A prospective cohort study was conducted in Leeds, UK.\"\n"
        "2. Identify dependent clauses/modifiers:\n"
        "   - Modifier A: \"based on routinely collected data from a service\"\n"
        "   - Dependent clause B: \"that allowed patients with symptoms of lung cancer to request CXR\"\n"
        "3. Rewrite each as a standalone simple sentence:\n\n"
        "Output:\n"
        "S1 -> A prospective cohort study was conducted in Leeds, UK.\n"
        "S2 -> The study was based on routinely collected data from a service.\n"
        "S3 -> The service allowed patients with symptoms of lung cancer to request CXR.\n\n"
        "Example 2:\n"
        "Input:\n"
        "\"After the cells were treated with the drug, which had been synthesized in our lab, we measured the change in fluorescence using a spectrophotometer.\"\n\n"
        "Chain-of-Thought:\n"
        "1. Independent clause: \"We measured the change in fluorescence using a spectrophotometer.\"\n"
        "2. Dependent clauses/modifiers:\n"
        "   - Dependent clause A: \"After the cells were treated with the drug\"\n"
        "   - Modifier B: \"which had been synthesized in our lab\"\n"
        "3. Rewrite each as standalone simple sentences:\n\n"
        "Output:\n"
        "S1 -> We measured the change in fluorescence using a spectrophotometer.\n"
        "S2 -> The cells were treated with the drug.\n"
        "S3 -> The drug had been synthesized in our lab.\n\n"
        "Now apply the same process to this new sentence:\n"
        "Input: \"{sentence}\"\n\n"
        "***OUTPUT ONLY the simplified sentences, one per line in the form S1 -> ..., S2 -> ..., etc., and nothing else.***"
    )
}

PROMPT_COMPOUND = {
    'system': "You are an expert in sentence simplification. Your task is to convert compound sentences into simple sentences.",
    'user': (
        "Below is a process to convert a compound sentence into simple sentences. For each example, think step by step, "
        "then output only the simplified sentences in the form:\n"
        "S1 -> ... S2 -> ... ...\n"
        "one per line, and nothing else.\n\n"
        "Example 1:\n"
        "Input:\n"
        "\"Lung cancer stands prominently among the foremost contributors to human mortality, distinguished by its elevated fatality rate and the second-highest incidence rate among malignancies, and the metastatic dissemination of lung cancer stands as a primary determinant of its elevated mortality and recurrence rates.\"\n\n"
        "Chain-of-Thought:\n"
        "1. Identify the independent clauses:\n"
        "   - IC1: \"Lung cancer stands prominently among the foremost contributors to human mortality.\"\n"
        "   - IC2: \"The metastatic dissemination of lung cancer stands as a primary determinant of its elevated mortality and recurrence rates.\"\n"
        "2. Identify modifiers:\n"
        "   - Modifier: \"distinguished by its elevated fatality rate and the second-highest incidence rate among malignancies\" (modifies IC1)\n"
        "3. Rewrite all parts as simple, standalone sentences.\n\n"
        "Output:\n"
        "S1 -> Lung cancer stands prominently among the foremost contributors to human mortality.\n"
        "S2 -> It is distinguished by its elevated fatality rate and the second-highest incidence rate among malignancies.\n"
        "S3 -> The metastatic dissemination of lung cancer stands as a primary determinant of its elevated mortality and recurrence rates.\n\n"
        "Example 2:\n"
        "Input:\n"
        "\"Climate change accelerates the melting of polar ice, and rising sea levels threaten coastal communities around the world.\"\n\n"
        "Chain-of-Thought:\n"
        "1. Identify the independent clauses:\n"
        "   - IC1: \"Climate change accelerates the melting of polar ice.\"\n"
        "   - IC2: \"Rising sea levels threaten coastal communities around the world.\"\n"
        "2. No dependent clauses or modifiers.\n"
        "3. Rewrite each as a standalone simple sentence.\n\n"
        "Output:\n"
        "S1 -> Climate change accelerates the melting of polar ice.\n"
        "S2 -> Rising sea levels threaten coastal communities around the world.\n\n"
        "Now apply the same process to this new sentence:\n"
        "Input: \"{sentence}\"\n\n"
        "OUTPUT ONLY the simplified sentences, one per line in the form S1 -> ..., S2 -> ..., etc., and nothing else."
    )
}

PROMPT_COMPOUND_COMPLEX = {
    'system': "You are an expert in sentence simplification. Your task is to convert compound-complex sentences into simple sentences.",
    'user': (
        "Below is a process to split a compound-complex sentence into standalone simple sentences. Think step by step, then apply.\n\n"
        "Example 1:\n"
        "Input:\n"
        "\"Although lung cancer is the leading cause of US cancer-related deaths, lung cancer screening with a low radiation dose chest computed tomography scan is now standard of care for a high risk eligible population, and clinicians and surgeons must evaluate the trade-offs of benefits and harms, including the identification of many benign lung nodules, overdiagnosis, and complications.\"\n\n"
        "Chain-of-Thought:\n"
        "1. Dependent Clause (DC): \"Although lung cancer is the leading cause of US cancer-related deaths\"\n"
        "2. Independent Clause 1 (IC1): \"Lung cancer screening with a low-dose chest computed tomography scan is now standard of care for a high-risk eligible population\"\n"
        "3. Independent Clause 2 (IC2): \"Clinicians and surgeons must evaluate the trade-offs of benefits and harms\"\n"
        "4. Modifier list: \"including the identification of many benign lung nodules, overdiagnosis, and complications\"\n"
        "5. Rewrite into standalone simple sentences.\n\n"
        "Output:\n"
        "S1 -> Lung cancer is the leading cause of US cancer-related deaths.\n"
        "S2 -> Lung cancer screening with a low-dose chest computed tomography scan is now standard of care for a high risk eligible population.\n"
        "S3 -> Lung cancer screening is recommended for a high-risk, eligible population.\n"
        "S4 -> Clinicians and surgeons must evaluate the trade-offs of benefits and harms.\n"
        "S5 -> Evaluated trade-offs include the identification of many benign lung nodules.\n"
        "S6 -> Evaluated trade-offs include the risk of overdiagnosis.\n"
        "S7 -> Evaluated trade-offs include complications from lung-cancer screening.\n\n"
        "Example 2:\n"
        "Input:\n"
        "\"Although warmed by the sun, the fields remained dry, and farmers worried about the drought.\"\n\n"
        "Chain-of-Thought:\n"
        "1. Dependent Clause (DC): \"Although warmed by the sun\"\n"
        "2. Independent Clause 1 (IC1): \"The fields remained dry\"\n"
        "3. Independent Clause 2 (IC2): \"Farmers worried about the drought\"\n"
        "4. Rewrite into standalone simple sentences.\n\n"
        "Output:\n"
        "S1 -> The sun warmed the fields.\n"
        "S2 -> The fields remained dry.\n"
        "S3 -> Farmers worried about the drought.\n\n"
        "Now apply to this new sentence:\n"
        "Input: \"{sentence}\"\n\n"
        "OUTPUT ONLY the simplified sentences, one per line in the form S1 -> ..., S2 -> ..., etc., and nothing else."
    )
}

TYPE_TO_PROMPT = {
    "complex sentence": PROMPT_COMPLEX,
    "compound sentence": PROMPT_COMPOUND,
    "compound-complex sentence": PROMPT_COMPOUND_COMPLEX,
}

PASS_THROUGH = {"simple sentence", "incomplete sentence"}

def build_prompt(prompt_dict, sentence: str) -> str:
    return prompt_dict["system"] + "\n\n" + prompt_dict["user"].format(sentence=sentence)

def extract_simplified_sentences(text: str):
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    outs = []
    for ln in lines:
        if not ln.startswith("S"):
            continue
        sep = "->" if "->" in ln else None
        if not sep:
            continue
        parts = ln.split(sep, 1)
        if len(parts) != 2:
            continue
        sent = parts[1].strip(" :\t")
        if sent:
            outs.append(sent)
    return outs

def load_llm(model_name: str, device_index: int, four_bit: bool):
    quant = None
    if four_bit:
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            quant = None

    use_gpu = (device_index >= 0 and torch.cuda.is_available())
    device_map = "auto" if use_gpu else None

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", None) or getattr(tok, "sep_token", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=quant
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok)
    return pipe, tok

def main():
    ap = argparse.ArgumentParser(description="Simplify sentences using predicted_label. Simple/Incomplete pass through; others simplified per type.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path (exploded)")
    ap.add_argument("--text-col", default="Sentence", help="Column containing sentences (default: Sentence)")
    ap.add_argument("--label-col", default="predicted_label", help="Column with predicted label (default: predicted_label)")
    ap.add_argument("--id-cols", default="", help="Comma-separated columns to carry over")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model id or local path")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (-1 for CPU)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per sentence")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    ap.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    ap.add_argument("--four-bit", action="store_true", help="Use 4-bit quantization (if available)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Columns: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found. Columns: {list(df.columns)}")

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    keep_cols = [c for c in id_cols if c in df.columns]

    # Normalize labels
    labels_norm = df[args.label_col].astype(str).str.strip().str.lower().fillna("")
    sentences = df[args.text_col].astype(str).fillna("")

    # Split indices into pass-through and to-simplify
    passthrough_idx = labels_norm.isin(PASS_THROUGH).to_list()
    to_simplify_idx = [not b for b in passthrough_idx]

    rows = []

    # Pass-through first
    for idx, do_pass in enumerate(passthrough_idx):
        if not do_pass:
            continue
        original = sentences.iloc[idx]
        label = labels_norm.iloc[idx]
        meta = {c: df.iloc[idx][c] for c in keep_cols}
        row = dict(meta)
        row["TypeUsed"] = f"{label} (pass-through)"
        row["OriginalSentence"] = original
        row["SimpleIndex"] = 1
        row["SimpleSentence"] = original
        rows.append(row)

    # Prepare simplification batches grouped by type
    simplify_indices = [i for i, b in enumerate(to_simplify_idx) if b]
    if simplify_indices:
        pipe, tok = load_llm(args.model, args.device, args.four_bit)

        # Group by normalized label
        sub_df = df.iloc[simplify_indices].copy()
        sub_labels = labels_norm.iloc[simplify_indices].copy()
        for norm_label, group_df in sub_df.groupby(sub_labels):
            prompt_dict = TYPE_TO_PROMPT.get(norm_label, PROMPT_COMPLEX)  # default to complex
            group_idxs = group_df.index.tolist()

            # Batch generation for this group
            for i in tqdm(range(0, len(group_idxs), args.batch_size), desc=f"Simplifying [{norm_label}]"):
                batch_idx = group_idxs[i:i+args.batch_size]
                batch_sents = [sentences.iloc[j] for j in batch_idx]
                prompts = [build_prompt(prompt_dict, s) for s in batch_sents]

                gen = pipe(
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=min(args.batch_size, len(batch_sents)),
                    truncation=True,
                    return_full_text=False,
                    pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=True,
                )

                # Normalize output structure
                outs = []
                for item in gen:
                    if isinstance(item, list) and item and isinstance(item[0], dict) and "generated_text" in item[0]:
                        outs.append(item[0]["generated_text"])
                    elif isinstance(item, dict) and "generated_text" in item:
                        outs.append(item["generated_text"])
                    else:
                        outs.append("")

                for j, out_text in enumerate(outs):
                    sims = extract_simplified_sentences(out_text)
                    original = batch_sents[j]
                    meta = {c: df.loc[batch_idx[j], c] for c in keep_cols}
                    if sims:
                        for k, s in enumerate(sims, start=1):
                            row = dict(meta)
                            row["TypeUsed"] = norm_label
                            row["OriginalSentence"] = original
                            row["SimpleIndex"] = k
                            row["SimpleSentence"] = s
                            rows.append(row)
                    else:
                        row = dict(meta)
                        row["TypeUsed"] = norm_label
                        row["OriginalSentence"] = original
                        row["SimpleIndex"] = 1
                        row["SimpleSentence"] = ""
                        rows.append(row)

    out_df = pd.DataFrame(rows, columns=(keep_cols + ["TypeUsed", "OriginalSentence", "SimpleIndex", "SimpleSentence"]))
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_path}")

if __name__ == "__main__":
    main()
