import argparse
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

PROMPT_TEMPLATE = {
    'system': """You are an expert in sentence simplification. Your task is to convert complex sentences into simple sentences.""",
    'user': """Below is a step-by-step process. For each example, think step by step, then output only the simplified sentences in the form:
S1 → ... S2 → ... ...
one per line, and nothing else.

Example 1:
Input:
"A prospective cohort study was conducted in Leeds, UK, based on routinely collected data from a service that allowed patients with symptoms of lung cancer to request CXR."

Chain-of-Thought:
1. Identify the independent clause: "A prospective cohort study was conducted in Leeds, UK."
2. Identify dependent clauses/modifiers:
   • Modifier A: "based on routinely collected data from a service"
   • Dependent clause B: "that allowed patients with symptoms of lung cancer to request CXR"
3. Rewrite each as a standalone simple sentence:

Output:
• S1 → A prospective cohort study was conducted in Leeds, UK.
• S2 → The study was based on routinely collected data from a service.
• S3 → The service allowed patients with symptoms of lung cancer to request CXR.

Example 2:
Input:
"After the cells were treated with the drug, which had been synthesized in our lab, we measured the change in fluorescence using a spectrophotometer."

Chain-of-Thought:
1. Independent clause: "We measured the change in fluorescence using a spectrophotometer."
2. Dependent clauses/modifiers:
   • Dependent clause A: "After the cells were treated with the drug"
   • Modifier B: "which had been synthesized in our lab"
3. Rewrite each as standalone simple sentences:

Output:
• S1 → We measured the change in fluorescence using a spectrophotometer.
• S2 → The cells were treated with the drug.
• S3 → The drug had been synthesized in our lab.

Now apply the same process to this new sentence:
Input: "{sentence}"

***OUTPUT ONLY the simplified sentences, one per line in the form S1 → ..., S2 → ..., etc., and nothing else.***"""
}

def build_prompt(sentence: str) -> str:
    return PROMPT_TEMPLATE["system"] + "\n\n" + PROMPT_TEMPLATE["user"].format(sentence=sentence)

def parse_simplified(output: str):
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    simples = []
    for ln in lines:
        if ln[:1].upper() == "S" and ("->" in ln or "→" in ln):
            sep = "->" if "->" in ln else "→"
            try:
                _, tail = ln.split(sep, 1)
                s = tail.strip().lstrip(":").strip()
                if s:
                    simples.append(s)
            except ValueError:
                continue
        else:
            if len(ln.split()) >= 2:
                simples.append(ln)
    seen = set()
    uniq = []
    for s in simples:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def load_model(model_name: str, device_index: int, four_bit: bool):
    device = torch.device(f"cuda:{device_index}") if (device_index >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    quant_cfg = None
    if four_bit and HAS_BNB:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) else tok.sep_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=quant_cfg,
        torch_dtype=getattr(torch, "bfloat16", None) if device.type == "cuda" else None,
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok, device=0 if device.type == "cuda" else -1)
    return pipe, tok

def run_inference(pipe: TextGenerationPipeline, tokenizer, sentences: List[str], batch_size: int, max_new_tokens: int, temperature: float, top_p: float):
    outputs_all = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Simplifying"):
        batch = sentences[i:i+batch_size]
        prompts = [build_prompt(s) for s in batch]
        gen = pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=min(batch_size, len(batch)),
            truncation=True,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        for out in gen:
            text = out[0]["generated_text"].strip() if isinstance(out, list) else out["generated_text"].strip()
            outputs_all.append(parse_simplified(text))
    return outputs_all

def main():
    ap = argparse.ArgumentParser(description="Simplify complex sentences in a CSV column into multiple simple sentences (LLM).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    ap.add_argument("--text-col", default="Sentence", help="Column containing sentences (default: Sentence)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path (exploded)")
    ap.add_argument("--id-cols", default="", help="Comma-separated columns to carry over")
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="HF model id or local path")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (-1 for CPU)")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per sentence")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    ap.add_argument("--four-bit", action="store_true", help="Use 4-bit quantization (requires bitsandbytes)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    keep_cols = [c for c in id_cols if c in df.columns]

    sentences = df[args.text_col].astype(str).fillna("").tolist()

    pipe, tok = load_model(args.model, args.device, args.four_bit)
    outputs = run_inference(pipe, tok, sentences, args.batch_size, args.max_new_tokens, args.temperature, args.top_p)

    rows = []
    for idx, sims in enumerate(outputs):
        if not sims:
            rows.append({**{c: df.loc[idx, c] for c in keep_cols}, "OriginalSentence": sentences[idx], "SimpleIndex": 1, "SimpleSentence": ""})
        else:
            for j, s in enumerate(sims, start=1):
                row = {c: df.loc[idx, c] for c in keep_cols}
                row["OriginalSentence"] = sentences[idx]
                row["SimpleIndex"] = j
                row["SimpleSentence"] = s
                rows.append(row)

    out_df = pd.DataFrame(rows, columns=(keep_cols + ["OriginalSentence", "SimpleIndex", "SimpleSentence"]))
    out_df.to_csv(args.out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_path}")

if __name__ == "__main__":
    main()
