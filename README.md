# Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs

Artifacts for the paper **“Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs,”** published in the **2025 IEEE International Conference on Big Data (BigData)**, pages 7943–7952.

- [IEEE Xplore](https://ieeexplore.ieee.org/document/11400857)
- [DOI: 10.1109/BigData66926.2025.11400857](https://doi.org/10.1109/BigData66926.2025.11400857)
- [Paper PDF](paper/Domain-Specific_Knowledge_Graphs_in_RAG-Enhanced_Healthcare_LLMs.pdf)

## Repository contents

The `artifacts/` directory contains the available project files:

- Python scripts for abstract tokenization, coreference processing, sentence splitting and tagging, sentence simplification, causal filtering, relation extraction, and local RAG evaluation.
- Selected, coreference-processed, sentence-level, labeled, and causal-filtered CSV data.
- Three knowledge-graph CSV files in `artifacts/questions/`:
  - `G1.csv` — 6,439 rows
  - `G2.csv` — 9,385 rows
  - `G3_from_triples_AND_columns.csv` — 8,238 rows
- Two multiple-choice probe files in `artifacts/questions/`:
  - `questions_final_schema_balanced.json` — 100 questions
  - `questions_final_schema_balanced2.json` — 110 questions
- Local evaluation outputs under these directories:
  - `Mixtral_T0`
  - `llama3T0`, `llama3T0.2`, and `llama3T0.5`
  - `llama3_2_T0`, `llama3_2_T0.2`, and `llama3_2_T0.5`

Each available evaluation directory contains prediction and metric files together with TF-IDF vectorizers and sparse matrices.

## Citation

```bibtex
@inproceedings{anuyah2025domain,
  title={Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs},
  author={Anuyah, Sydney and Kaushik, Mehedi Mahmud and Dai, Hao and Shiradkar, Rakesh and Durresi, Arjan and Chakraborty, Sunandan},
  booktitle={2025 IEEE International Conference on Big Data (BigData)},
  pages={7943--7952},
  year={2025},
  organization={IEEE}
}
```
