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

---

## Data

Raw PubMed exports are not tracked in the repository, but the processed CSV files are stored under `data_pre_processing/` for convenience.

Main intermediate files:

- `full_symbiosis_articles.csv` – combined raw/parsed PubMed records.
- `cleaned_symbiosis_articles.csv` – cleaned text (lowercased, tokenized/normalized).
- `full_cleaned_symbiosis_articles.csv` – alternative cleaned version used for some experiments.
- `labeled_symbiosis_data.csv` – final labeled dataset (text + functional label).
- `symbiosis_articles.csv` – additional processed view of the corpus used in some analyses.

### Typical data flow

1. **Collect raw abstracts**

        python python_scripts/full_data_collection.py \
            --output data_pre_processing/full_symbiosis_articles.csv

2. **Preprocess text**

        python python_scripts/data_preprocessor.py \
            --input  data_pre_processing/full_symbiosis_articles.csv \
            --output data_pre_processing/cleaned_symbiosis_articles.csv

3. **Label abstracts**

        python python_scripts/data_labeling.py \
            --input  data_pre_processing/cleaned_symbiosis_articles.csv \
            --output data_pre_processing/labeled_symbiosis_data.csv

4. **(Optional) Build train/validation/test splits**

        python python_scripts/data_processor.py \
            --input  data_pre_processing/labeled_symbiosis_data.csv \
            --train  data_pre_processing/symbiosis_train.csv \
            --val    data_pre_processing/symbiosis_val.csv \
            --test   data_pre_processing/symbiosis_test.csv

If the CSV files are already present in `data_pre_processing/`, you can skip steps 1–4 and go straight to model training.

---

## Training

You can fine-tune the BioBERT classifier locally using the processed CSV files in `data_pre_processing/`:

    python python_scripts/finetune_model.py \
        --train_file data_pre_processing/symbiosis_train.csv \
        --val_file   data_pre_processing/symbiosis_val.csv \
        --model_name dmis-lab/biobert-base-cased-v1.1 \
        --output_dir models/biobert_symbiosis \
        --epochs 5 \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --max_length 256

If you run on a cluster with SLURM (or a similar scheduler), you can submit the pre-configured job script:

    sbatch jobs/finetune_job.sh

The script reports standard metrics on the validation split (accuracy, precision, recall, F1) and saves the fine-tuned model under `models/biobert_symbiosis/`.

---

## Inference on new abstracts

To classify a new abstract from the command line:

    python python_scripts/predict_gene.py \
        --model_dir models/biobert_symbiosis \
        --text "In this study we investigate quorum sensing in Vibrio fischeri..."

Example output:

    Predicted label: quorum_sensing
    Probabilities:
      quorum_sensing    0.91
      motility          0.03
      biofilm           0.04
      host_interaction  0.02

On a cluster, you can use the job script instead:

    sbatch jobs/predict_job.sh

---

## Topic modeling

Besides supervised classification, the project includes an LDA topic-modeling step to explore latent themes in the symbiosis literature:

    python python_scripts/topic_modeling.py \
        --input      data_pre_processing/cleaned_symbiosis_articles.csv \
        --output_dir python_scripts/lda_vis/

The resulting interactive HTML visualisation (`python_scripts/lda_vis/lda_visualization.html`) can be opened in a browser to inspect topics and their most representative terms.

---

## Citation

If you use this code or ideas from this repository in your work, please consider citing:

> Pérez, P. (2025). *Symbiosis Abstract LLM: fine-tuning BioBERT for multi-class classification of symbiosis-related scientific abstracts.* GitHub repository.

