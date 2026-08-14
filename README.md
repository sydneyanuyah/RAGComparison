# Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs

Artifacts for the paper **“Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs,”** published in the **2025 IEEE International Conference on Big Data (BigData)**, pages 7943–7952.

- [IEEE Xplore](https://ieeexplore.ieee.org/document/11400857)
- [DOI: 10.1109/BigData66926.2025.11400857](https://doi.org/10.1109/BigData66926.2025.11400857)
- [Paper PDF](paper/Domain-Specific_Knowledge_Graphs_in_RAG-Enhanced_Healthcare_LLMs.pdf)

## Repository contents

### Project artifacts

The `artifacts/` directory contains the available project files collected from the project server and local project folders:

- Python scripts for data selection, abstract tokenization, coreference processing, sentence splitting and tagging, sentence simplification, causal filtering, relation extraction, and local RAG evaluation.
- Raw, selected, tokenized, coreference-processed, sentence-level, labeled, and causal-filtered CSV data.
- Three knowledge-graph CSV files:
  - `G1.csv` - 6,439 rows
  - `G2.csv` - 9,385 rows
  - `G3_from_triples_AND_columns.csv` - 8,238 rows
- Two multiple-choice probe files:
  - `questions_final_schema_balanced.json` - 100 questions
  - `questions_final_schema_balanced2.json` - 110 questions

`artifacts/local_ieee_snapshot/` preserves the available local project snapshot. It includes:

- `RawData.csv` - 27,973 PubMed records
- `selected_AD.csv`, `selected_T2DM.csv`, and `selected_ALZ_T2DM.csv` - 1,000 selected records each
- `ieee_datafilter.py` and the available processing scripts
- Tokenized datasets, labeled sentence data, causal-filtered data, relation-extraction outputs, and `re_task.zip`

`artifacts/local_core/` preserves the locally stored graph files, probe files, and RAG evaluation script.

### Evaluation outputs

`artifacts/questions/` contains the evaluation outputs available on the project server.

`artifacts/local_results/` contains 22 complete local evaluation bundles under the directory names in which they were found:

- `Qwen2.5T0`, `Qwen2.5T0.2`, and `Qwen2.5T0.5`
- `Qwen2.5_2_T0`, `Qwen2.5_2_T0.2`, and `Qwen2.5_2_T0.5`
- `Mixtral2.5T0`, `Mixtral2.5T0.2`, and `Mixtral2.5T0.5`
- `Mixtral2.5_2_T0`, `Mixtral2.5_2_T0.2`, and `Mixtral2.5_2_T0.5`
- `Mistral2.5T0.2` and `Mistral2.5T0.5`
- `Mistral2.5_2_T0`, `Mistral2.5_2_T0.2`, and `Mistral2.5_2_T0.5`
- `llama3T0`, `llama3T0.2`, and `llama3T0.5`
- `llama3_2_T0` and `llama3_2_T0.2`

Each complete evaluation bundle contains predictions, CSV and JSON metrics, five TF-IDF vectorizers, and five sparse matrices. The available server outputs also include the `llama3_2_T0.5` bundle.

### Paper

`paper/Domain-Specific_Knowledge_Graphs_in_RAG-Enhanced_Healthcare_LLMs.pdf` is the published IEEE paper included with this repository.

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
