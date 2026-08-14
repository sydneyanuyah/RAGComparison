import os
import re
import json
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SEED = 4000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FICL_PROMPT = {
    "system": """You are a coreference resolution agent. Below is a biomedical abstract presented as tokenized text with indices. Your task is to identify and annotate coreference expressions within the text.

Use this format:
[
  {
    "Expression": "string",
    "StartToken": int,
    "EndToken": int,
    "RefersTo": "string"
  }
]

Example:
Given this tokenized abstract:
("BACKGROUND:", 0), ("There", 1), ("are", 2), ("few", 3), ("cases", 4), ("of", 5), ("pulmonary", 6), ("granulomatous", 7), ("changes", 8), ("secondary", 9), ("to", 10), ("primary", 11), ("biliary", 12), ("cirrhosis", 13), ("(PBC).", 14), ("No", 15), ("case", 16), ("of", 17), ("granulomatous", 18), ("lung", 19), ("disease", 20), ("secondary", 21), ("to", 22), ("PBC", 23), ("misdiagnosed", 24), ("as", 25), ("lung", 26), ("cancer", 27), ("had", 28), ("been", 29), ("reported.", 30), ("CASE", 31), ("SUMMARY:", 32), ("A", 33), ("middle-aged", 34), ("woman", 35), ("presented", 36), ("with", 37), ("lung", 38), ("nodules", 39), ("and", 40), ("was", 41), ("misdiagnosed", 42), ("with", 43), ("lung", 44), ("cancer", 45), ("by", 46), ("positron", 47), ("emission", 48), ("tomography/computed", 49), ("tomography.", 50), ("She", 51), ("underwent", 52), ("left", 53), ("lobectomy,", 54), ("and", 55), ("the", 56), ("pathology", 57), ("of", 58), ("the", 59), ("nodules", 60), ("showed", 61), ("granulomatous", 62), ("inflammation,", 63), ("which", 64), ("was", 65), ("then", 66), ("treated", 67), ("with", 68), ("antibiotics.", 69), ("However,", 70), ("a", 71), ("new", 72), ("nodule", 73), ("appeared.", 74), ("Further", 75), ("investigation", 76), ("with", 77), ("lung", 78), ("biopsy", 79), ("and", 80), ("liver", 81), ("serology", 82), ("led", 83), ("to", 84), ("the", 85), ("diagnosis", 86), ("of", 87), ("PBC,", 88), ("and", 89), ("chest", 90), ("computed", 91), ("tomography", 92), ("indicated", 93), ("significant", 94), ("reduction", 95), ("in", 96), ("the", 97), ("pulmonary", 98), ("nodule", 99), ("by", 100), ("treatment", 101), ("with", 102), ("methylprednisolone", 103), ("and", 104), ("ursodeoxycholic", 105), ("acid.", 106), ("CONCLUSION:", 107), ("Diagnosis", 108), ("of", 109), ("pulmonary", 110), ("nodules", 111), ("requires", 112), ("integrating", 113), ("various", 114), ("clinical", 115), ("data", 116), ("to", 117), ("avoid", 118), ("unnecessary", 119), ("pulmonary", 120), ("lobectomy.", 121)

[
  {"Expression": "PBC", "StartToken": 14, "EndToken": 14, "RefersTo": "Primary biliary cirrhosis"},
  {"Expression": "PBC", "StartToken": 23, "EndToken": 23, "RefersTo": "Primary biliary cirrhosis"},
  {"Expression": "She", "StartToken": 51, "EndToken": 51, "RefersTo": "A middle-aged woman"},
  {"Expression": "PBC", "StartToken": 88, "EndToken": 88, "RefersTo": "Primary biliary cirrhosis"}
]""",
    "user": "Now process this tokenized abstract:\n\n{tokenized_text}"
}

def build_prompt(template, tokenized_text):
    return template['system'] + "\n\n" + template['user'].format(tokenized_text=tokenized_text)

def extract_coreference_json(text):
    try:
        first_bracket = text.find('[')
        last_bracket = text.rfind(']')
        if first_bracket == -1 or last_bracket == -1:
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace == -1 or last_brace == -1:
                return [], f"No valid JSON structure found: {text}"
            json_str = f"[{text[first_brace:last_brace+1]}]"
        else:
            json_str = text[first_bracket:last_bracket+1]
        json_str = json_str.replace(',]', ']').replace(',}', '}')
        obj = json.loads(json_str)
        if isinstance(obj, list) and all(isinstance(item, dict) and 'Expression' in item for item in obj):
            return obj, ""
        return [], f"Invalid JSON list format: {json_str}"
    except Exception as e:
        return [], f"Malformed JSON: {e} | Raw: {text}"

def parse_tokenized_text(tokenized_text):
    pattern = r'\("([^"]*)",\s*(\d+)\)'
    matches = re.findall(pattern, tokenized_text)
    return [(token, int(idx)) for token, idx in matches]

def switch_coreferences(tokenized_text, coreferences):
    tokens = parse_tokenized_text(tokenized_text)
    sorted_coref = sorted(coreferences, key=lambda x: x.get('StartToken', 10**12))
    replacements = {}
    for c in sorted_coref:
        try:
            s = int(c['StartToken']); e = int(c['EndToken']); ref = str(c['RefersTo']).strip()
        except Exception:
            continue
        for i in range(s, e+1):
            replacements[i] = ref if i == s else None
    result_tokens = []
    for token, idx in tokens:
        if idx in replacements:
            if replacements[idx] is not None:
                result_tokens.append(replacements[idx])
        else:
            result_tokens.append(token)
    joined = ' '.join(result_tokens)
    joined = re.sub(r'\s+([.,;:!?])', r'\1', joined)
    joined = re.sub(r'\(\s+', '(', joined)
    joined = re.sub(r'\s+\)', ')', joined)
    joined = re.sub(r'\s+', ' ', joined)
    return joined.strip()

def load_model(model_name: str):
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    print(f"Loading tokenizer for {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Loading model {model_name}...")
    mdl = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    return tok, mdl

def generate_coref(tokenizer, model, tokenized_text, max_new=1024, temperature=0.7, top_k=40):
    prompt = build_prompt(FICL_PROMPT, tokenized_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    coref, error = extract_coreference_json(response)
    return response, coref, error

def main():
    ap = argparse.ArgumentParser(description="Run coreference on token-index Abstracts and write switched abstracts.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path (with Abstract as token-index pairs)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HF model id")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=None, help="Only process first N rows")
    ap.add_argument("--start-row", type=int, default=0, help="Start row offset")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.in_path, encoding=args.encoding, engine="python", on_bad_lines="skip")
    except Exception as e:
        raise SystemExit(f"ERROR reading CSV: {e}")

    if "Abstract" not in df.columns:
        raise SystemExit("ERROR: Column 'Abstract' not found. Expect token-index pairs in this column.")

    if "OldAbstract" not in df.columns:
        df["OldAbstract"] = df["Abstract"]

    tokenizer, model = load_model(args.model)

    start = max(0, args.start_row)
    end = len(df) if args.max_rows is None else min(len(df), start + args.max_rows)

    coref_json_list = []
    coref_error_list = []
    switched_list = []

    for i in range(len(df)):
        if i < start or i >= end:
            coref_json_list.append("")
            coref_error_list.append("skipped")
            switched_list.append(df.loc[i, "OldAbstract"] if "OldAbstract" in df.columns else df.loc[i, "Abstract"])
            continue

        tokenized_text = str(df.loc[i, "Abstract"])
        try:
            raw, coref, err = generate_coref(tokenizer, model, tokenized_text, max_new=args.max_new,
                                             temperature=args.temperature, top_k=args.top_k)
            if err:
                coref_json_list.append("")
                coref_error_list.append(err)
                switched_list.append(df.loc[i, "OldAbstract"] if "OldAbstract" in df.columns else df.loc[i, "Abstract"])
            else:
                switched = switch_coreferences(tokenized_text, coref)
                coref_json_list.append(json.dumps(coref, ensure_ascii=False))
                coref_error_list.append("")
                switched_list.append(switched)
        except Exception as e:
            coref_json_list.append("")
            coref_error_list.append(f"generation failed: {e}")
            switched_list.append(df.loc[i, "OldAbstract"] if "OldAbstract" in df.columns else df.loc[i, "Abstract"])

        if (i - start + 1) % 5 == 0 and start <= i < end:
            print(f"Processed {i - start + 1} rows")

    df["CorefJSON"] = coref_json_list
    df["CorefError"] = coref_error_list
    df["Abstract"] = switched_list

    try:
        df.to_csv(args.out_path, index=False, encoding=args.encoding)
    except Exception as e:
        raise SystemExit(f"ERROR writing CSV: {e}")

    print(f"Wrote {len(df)} rows -> {args.out_path}")

if __name__ == "__main__":
    main()
