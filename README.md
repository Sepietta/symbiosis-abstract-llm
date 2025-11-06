# Symbiosis Abstract LLM

A specialized NLP pipeline to classify **scientific abstracts on symbiosis**.  
This project fine-tunes a **BioBERT**-based model on a custom-curated dataset of PubMed abstracts and performs **multi-class text classification** into biologically meaningful categories (e.g. `quorum_sensing`, `motility`, `biofilm`, `host_interaction`, etc.).

---

## Project goals

- Build a **curated dataset** of PubMed abstracts related to microbial/host symbiosis.
- Clean and preprocess raw text (lowercasing, tokenization, stopword handling, etc.).
- Fine-tune a **BioBERT** model for multi-class text classification.
- Evaluate performance with accuracy, F1-score and confusion matrices.
- Provide a **reproducible pipeline** that can be adapted to other biological topics.

---

## Repository structure

```text
symbiosis-abstract-llm/
├─ data_pre_processing/
│  ├─ ...                # scripts/notebooks to download and clean abstracts
│  └─ ...
├─ jobs/
│  ├─ ...                # job scripts for cluster / SLURM / bash execution
│  └─ ...
├─ nltk_data/
│  └─ corpora/           # NLTK resources used during preprocessing (can be generated locally)
├─ python_scripts/
│  ├─ dataset_builder.py # builds the labeled dataset from cleaned abstracts
│  ├─ train_model.py     # main training script for the BioBERT model
│  ├─ evaluate_model.py  # evaluation on the held-out test set
│  └─ infer_single.py    # classify a new abstract from the command line
├─ .gitignore
└─ README.md
