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
│  ├─ full_symbiosis_articles.csv      # raw/combined PubMed articles
│  ├─ cleaned_symbiosis_articles.csv   # cleaned text
│  ├─ symbiosis_articles.csv           # intermediate dataset
│  ├─ labeled_symbiosis_data.csv       # labeled abstracts (text + label)
│  └─ symbiosis_articles_test.csv      # (example) split used for testing
├─ jobs/
│  ├─ finetune_job.sh                  # job script to fine-tune the model on a cluster
│  └─ predict_job.sh                   # job script to run prediction on a cluster
├─ nltk_data/
│  └─ corpora/
│     ├─ stopwords/                    # NLTK stopwords (unzipped)
│     ├─ stopwords.zip                 # zipped stopwords
│     └─ wordnet.zip                   # zipped WordNet corpus
├─ python_scripts/
│  ├─ lda_vis/
│  │  └─ lda_visualization.html        # interactive LDA visualization
│  ├─ data_labeling.py                 # assign functional labels to abstracts
│  ├─ data_preprocessor.py             # text cleaning (lowercasing, tokenization, etc.)
│  ├─ data_processor.py                # build final train/val/test CSVs
│  ├─ finetune_model.py                # fine-tune the BioBERT classifier
│  ├─ full_data_collection.py          # bulk download of symbiosis-related PubMed abstracts
│  ├─ predict_gene.py                  # inference on new abstracts (predict class)
│  └─ topic_modeling.py                # LDA topic modeling on the corpus
├─ .gitignore
└─ README.md
