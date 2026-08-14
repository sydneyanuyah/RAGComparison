# ieee_datafilter.py
import os
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CAUSALITY_TERMS = [
    "causal","causality","cause","effect","effects","mediate","mediation",
    "mechanism","mechanistic","pathway","risk factor","determinant",
    "impact","influence","lead to","result in","Mendelian randomization",
    "randomized","intervention","trial","longitudinal","temporal precedence",
    "instrumental variable"
]
PHENOTYPE_TERMS = [
    "phenotype","phenotypic","clinical feature","clinical features",
    "symptom","symptoms","presentation","subtype","endophenotype",
    "cognitive decline","memory impairment","cognition","neuropsychiatric",
    "neurological","metabolic phenotype","insulin resistance phenotype",
    "complication","disease severity","progression"
]
BIOMARKER_TERMS = [
    "biomarker","marker","blood","plasma","serum","csf","cerebrospinal fluid",
    "tau","phosphorylated tau","p-tau","t-tau","amyloid","aβ","abeta","a-beta",
    "beta-amyloid","amyloid-beta","insulin","glucose","hba1c",
    "apoe","apoe4","apoe ε4","snp","gwas","proteomic","metabolomic",
    "transcriptomic","genomic","cytokine","il-6","tnf","crp","neurofilament light",
    "nfl","phospho-tau","p-tau-181","p-tau-217"
]

CAUSALITY_QUERY = " ".join(CAUSALITY_TERMS)
PHENOTYPE_QUERY = " ".join(PHENOTYPE_TERMS)
BIOMARKER_QUERY = " ".join(BIOMARKER_TERMS)

def _word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))

def _keyword_hits(text: str, terms) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    hits = 0
    lowered = text.lower()
    for t in terms:
        pattern = r"\b" + re.escape(t.lower()).replace(r"\ ", r"\s+") + r"\b"
        hits += len(re.findall(pattern, lowered))
    return hits

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    if math.isclose(max_v, min_v):
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)

def select_top_abstracts(
    df: pd.DataFrame,
    per_group: int = 1000,
    text_col_title: str = "Title",
    text_col_abs: str = "Abstract",
    group_col: str = "csv_name",
    groups_expected=("AD","T2DM","ALZ_T2DM"),
    weights=(0.25,0.25,0.25,0.25)  # (causal_sim, pheno_sim, biomarker_sim, keyword_boost)
) -> dict:
    assert len(weights) == 4, "weights must be a 4-tuple"
    w_causal, w_pheno, w_biom, w_kw = weights

    for col in [text_col_title, text_col_abs, group_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["__AbstractWordCount"] = df[text_col_abs].apply(_word_count)
    df_filt = df[df["__AbstractWordCount"] >= 180].copy()

    df_filt["__text"] = (
        df_filt[text_col_title].fillna("").astype(str).str.strip() + ". " +
        df_filt[text_col_abs].fillna("").astype(str).str.strip()
    ).str.strip()

    # If very small groups, set min_df=1 to avoid empty vocab
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(df_filt["__text"])

    q_causal = vectorizer.transform([CAUSALITY_QUERY])
    q_pheno  = vectorizer.transform([PHENOTYPE_QUERY])
    q_biom   = vectorizer.transform([BIOMARKER_QUERY])

    sim_causal = cosine_similarity(X, q_causal).ravel()
    sim_pheno  = cosine_similarity(X, q_pheno).ravel()
    sim_biom   = cosine_similarity(X, q_biom).ravel()

    kw_causal = df_filt["__text"].apply(lambda t: _keyword_hits(t, CAUSALITY_TERMS)).to_numpy()
    kw_pheno  = df_filt["__text"].apply(lambda t: _keyword_hits(t, PHENOTYPE_TERMS)).to_numpy()
    kw_biom   = df_filt["__text"].apply(lambda t: _keyword_hits(t, BIOMARKER_TERMS)).to_numpy()
    kw_total  = kw_causal + kw_pheno + kw_biom

    n_causal = _normalize(sim_causal)
    n_pheno  = _normalize(sim_pheno)
    n_biom   = _normalize(sim_biom)
    n_kw     = _normalize(kw_total)

    combined = w_causal*n_causal + w_pheno*n_pheno + w_biom*n_biom + w_kw*n_kw

    df_filt["Score_CausalSim"] = n_causal
    df_filt["Score_PhenoSim"]  = n_pheno
    df_filt["Score_BiomSim"]   = n_biom
    df_filt["Score_Keywords"]  = n_kw
    df_filt["Score_Combined"]  = combined

    results = {}
    available_groups = df_filt[group_col].dropna().astype(str).unique().tolist()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for g in groups_expected:
        if g not in available_groups:
            continue
        sub = df_filt[df_filt[group_col] == g].copy()
        sub = sub.sort_values("Score_Combined", ascending=False).head(per_group)
        keep_cols = [c for c in df.columns if not c.startswith("__")]
        keep_cols += ["Score_CausalSim","Score_PhenoSim","Score_BiomSim","Score_Keywords","Score_Combined"]
        sub = sub[[c for c in keep_cols if c in sub.columns]]

        out_path = os.path.join(out_dir, f"selected_{g}.csv")
        sub.to_csv(out_path, index=False)
        results[g] = {"df": sub, "path": out_path, "count": len(sub)}

    return results

if __name__ == "__main__":
    # Change "RawData.csv" to your actual file if different.
    in_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RawData.csv")
    df = pd.read_csv(in_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    res = select_top_abstracts(df)
    for g, r in res.items():
        print(f"{g}: wrote {r['count']} rows -> {r['path']}")
