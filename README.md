# COS-802-Project
Semantic Textual Similarity (STS) Across Domains and Languages

**Author:** Linda Masia

## Overview

This project looks at how well pretrained sentence embeddings handle semantic textual similarity (STS) when the text domain changes. It compares three encoder families on four STS benchmarks from four domains:

| Domain       | Dataset  | Split used for evaluation |
|--------------|----------|---------------------------|
| General      | STS-B    | test                      |
| Biomedical   | BIOSSES  | train (the only split)    |
| Financial    | FinSTS   | test                      |
| Social media | PIT-2015 | test                      |

Encoders:

* **GloVe**: mean of `glove-wiki-gigaword-50` word vectors
* **BERT**: mean-pooled `bert-base-uncased` token embeddings
* **SBERT**: `all-MiniLM-L6-v2` from Sentence-Transformers

Research questions:

* **RQ1:** How well do pretrained sentence embeddings generalise across general, biomedical, financial and social media STS data?
* **RQ2:** How do text characteristics (mean sentence length, length difference, vocabulary overlap) relate to SBERT's prediction error?
* **RQ3:** How much does fine-tuning on small amounts of in-domain data (500 to 2,000 pairs) help on the financial and social media domains?
* **RQ3.5:** How does parameter-efficient fine-tuning (LoRA) compare with full fine-tuning on Spearman correlation, training time and peak GPU memory?
* **RQ4:** Do GloVe, BERT and SBERT differ in how sensitive they are to domain shift?

What sets it apart:

* All similarity scores are put on a common [0, 1] scale. This includes converting PIT-2015's crowd vote counts into graded similarity scores.
* Spearman correlations come with 95% bootstrap confidence intervals (1,000 resamples).
* Full fine-tuning and LoRA are compared on both models, with training time and peak GPU memory recorded.
* Processed datasets and results are cached, so the notebook can be re-run without reprocessing.

## Contents of the Zip File

```
COS802_STS_u21607363/
├── README.md               This file
└── COS802_STS.ipynb        Main notebook: data loading, EDA, zero-shot evaluation, RQ2 analysis,
                            full fine-tuning and LoRA experiments, plots (saved outputs included)
```

No datasets are included. They are all public and are downloaded automatically (see [Data Information](#data-information)).

## Setup Instructions

The project is meant to run in **Google Colab**, which is where the results were produced. Nothing needs installing on your own machine.

1. Go to <https://colab.research.google.com> and sign in with a Google account.
2. Upload the notebook: *File → Upload notebook*, then choose `COS802_STS.ipynb` from the unzipped folder.
3. Switch to a GPU runtime: *Runtime → Change runtime type*, choose **T4 GPU**, and save. The original runs used a T4 with the high-RAM setting.
4. Libraries are handled by the notebook. Colab already includes PyTorch, Transformers, pandas and the plotting libraries, and the notebook's `%pip install` cells add the rest (`sentence-transformers`, `datasets`, `gensim`, `peft`, `torchao`). The library versions tested are listed under [Software Versions](#software-versions).
5. When the first code cell runs, Colab asks for permission to connect your Google Drive. Allow it. Processed datasets and result tables are saved to `MyDrive/COS802_STS/` so that later runs can reuse them.

## Software Versions

Tested with Python 3.12 and these library versions:

| Library | Version |
|---------|---------|
| torch | 2.15.0 |
| transformers | 5.24.0 |
| sentence-transformers | 6.3.0 |
| datasets | 5.2.0 |
| accelerate | 1.16.0 |
| peft | 0.22.0 |
| torchao | 0.18.0 |
| gensim | 4.4.0 |
| numpy | 2.5.5 |
| pandas | 3.1.1 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| statsmodels | 0.15.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| tqdm | 4.70.1 |

If a later Colab image causes a compatibility error, install the versions above in a new cell before running the notebook, for example `%pip install -q sentence-transformers==6.3.0 transformers==5.24.0 datasets==5.2.0`.

## Running the Code

All experiments are in `COS802_STS.ipynb`. There are no command-line arguments or configuration files.

1. Complete the setup steps above.
2. Choose *Runtime → Run all*.
3. Approve the Google Drive prompt when it appears. The notebook then downloads the datasets and models and runs every experiment from top to bottom.

If the runtime disconnects partway through, reconnect and choose *Run all* again. Cached datasets and RQ3 results are loaded from Google Drive instead of being recomputed.

Notebook sections:

| Section | What it does |
|---------|--------------|
| 1. Setup and config | Imports, Google Drive cache folder, model names, seeds (`SEED = 2157`), hyperparameters |
| 2. EDA | Downloads and normalises the four datasets, then plots dataset sizes, score distributions and sentence lengths |
| 3. Model | Defines the GloVe, BERT and SBERT encoders and the cosine-similarity scorer |
| 4. Zero-shot evaluation (RQ1, RQ4) | Spearman and Pearson correlations with bootstrap CIs, plus plots |
| RQ2 | Correlates text characteristics with SBERT's absolute error |
| RQ3 | Full fine-tuning of BERT and SBERT on 500 to 2,000 in-domain pairs (PIT-2015 train, FinSTS validation) |
| RQ3.5 | Same sweep with LoRA (r = 8, α = 16, query/value), with the time and memory comparison plots |
| Final cells | Reload saved result CSVs from the cache folder |

Configuration is done by editing variables in the first code cell:

| Variable | Default | Meaning |
|----------|---------|---------|
| `QUICK` | `False` | `True` uses fewer bootstrap samples (200 instead of 1,000) for a quick smoke test |
| `N_BOOT` | 1000 | Number of bootstrap resamples for the confidence intervals |
| `ADAPT_EPOCHS`, `ADAPT_LR`, `ADAPT_BS` | 3, 2e-5, 16 | Fine-tuning epochs, learning rate (LoRA uses 10×) and batch size |
| `SEED` | 2157 | Global random seed, also used to sample the training subsets |
| `BERT_NAME`, `SBERT_NAME`, `GLOVE_NAME` | see notebook | Models used |

Caching: the processed datasets and the RQ3 results are loaded from `MyDrive/COS802_STS/` if they already exist there. Delete that folder in Google Drive to recompute everything from scratch. The LoRA cell always recomputes (`FORCE_RECOMPUTE = True`).

Expected runtime on a Colab T4: zero-shot evaluation takes a few minutes. The fine-tuning sweeps (2 models × 2 domains × 7 training sizes each) take the longest. In the recorded run, full fine-tuning took about 40 minutes of training time and LoRA about 32 minutes.

Outputs: figures appear inline in the notebook. Result tables are written to `rq3_results_high_res_resources.csv` and `lora_results_high_res_resources.csv` in `MyDrive/COS802_STS/`.

## Data Information

All datasets are public and downloaded automatically by the notebook, so no manual setup is needed.

| Dataset | Source | Notes |
|---------|--------|-------|
| STS-B | Hugging Face: [`sentence-transformers/stsb`](https://huggingface.co/datasets/sentence-transformers/stsb) | Scores are already in [0, 1] |
| BIOSSES | Hugging Face: [`tabilab/biosses`](https://huggingface.co/datasets/tabilab/biosses) | 100 pairs; scores 0 to 4 are divided by 4 |
| FinSTS | Hugging Face: [`syang687/FinSTS`](https://huggingface.co/datasets/syang687/FinSTS) (raw rows also shown from [`FinanceMTEB/FinSTS`](https://huggingface.co/datasets/FinanceMTEB/FinSTS)) | Scores 0 to 5 are divided by 5; the validation split is used for fine-tuning |
| PIT-2015 | GitHub: [cocoxu/SemEval-PIT2015](https://github.com/cocoxu/SemEval-PIT2015) ([zip](https://raw.githubusercontent.com/cocoxu/SemEval-PIT2015/master/data/SemEval-PIT2015-github.zip)) | Train labels `(yes, no)` become yes/(yes+no); test labels (0 to 5 votes) are divided by 5 |

Pretrained models (downloaded on first use): [`bert-base-uncased`](https://huggingface.co/bert-base-uncased), [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and `glove-wiki-gigaword-50` via [gensim-data](https://github.com/RaRe-Technologies/gensim-data).

Where the data ends up (all inside the Colab session or your Google Drive):

* Hugging Face datasets and models, and the GloVe vectors: the Colab runtime's temporary cache, downloaded again in each new session
* Processed, normalised datasets and result tables: `MyDrive/COS802_STS/` in your Google Drive
