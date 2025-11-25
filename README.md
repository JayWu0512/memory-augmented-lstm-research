# 📘 Memory-Augmented-LSTM-Research  
*A Modular Long-Context Language Modeling Framework*

This repository contains a standalone implementation of the **Memory-Augmented LSTM** framework used in the SkillMiner project.  
It is fully separated from the SkillMiner backend so that long‑context reasoning research can be reproduced independently.

---

## 🔍 1. Motivation

Standard LSTM models suffer from well-known limitations in long-context understanding due to fixed hidden-state size and vanishing gradients.  
This repository introduces explicit **short-term** and **long-term** memory components to extend LSTM behavior beyond its default capacity.

The models in this repo implement:

- A **base LSTM** language model
- A **summarization layer** module
- A **token-limit controller** module
- A **NER-based memory** module
- A **semantic memory** module

These modules are designed based on experiments documented in the project report and serve as the research foundation for SkillMiner’s memory design.

---

## 📂 2. Repository Structure

Your repo currently looks like this:

```text
.
├── cache_logs/                 # json that stores full mempry inputs
├── cache_qa/                   # Cache for synthetic QA experiments
├── cache_qa_dog_cat/           # Cache for dog–cat QA experiments
├── checkpoints_qa/             # Model checkpoints for synthetic QA
├── checkpoints_qa_dog_cat/     # Model checkpoints for dog–cat QA
├── data/                       # CSV datasets (see below)
├── logs/                       # Training / evaluation logs
├── nlp_final_project/          # Report, figures, or notebooks 
│
├── base_lstm_model.py          # Base LSTM definition (shared by all models)
├── memory_features.py          # Memory utilities (STM/LTM, features, helpers)
│
├── model_0_base.py             # Model 0 – vanilla LSTM
├── model_1_summarization.py    # Model 1 – LSTM + summarization
├── model_2_sum_toklimit.py     # Model 2 – + token limit controller
├── model_3_sum_tok_ner.py      # Model 3 – + NER memory
├── model_4_full_memory.py      # Model 4 – + semantic memory
│
├── train_qa_memory.py          # Train / evaluate on synthetic SkillMiner-style QA
└── train_qa_dog_cat.py         # Train / evaluate on dog–cat QA dataset
```

---

## 📦 3. Installation & Special Setup

### 3.1 Python version

Tested with:

```text
Python 3.11
```

### 3.2 Install Python packages

Create an environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` could look like:

### 3.3 Download spaCy model (for NER)

Model 3 and Model 4 use spaCy NER. You must download a language model once:

```bash
python -m spacy download en_core_web_sm
```

### 3.4 OpenAI API key (for LLM-as-a-judge)

If you want to reproduce the **LLM-as-a-judge** evaluation:

```bash
cp .env.example .env
```
And then add your api key into .env file

---

## 📊 4. Datasets

### 4.1 Folder layout

All datasets are expected under `data/`:

```text
data/
    skillminer_qa_synthetic.csv    # main synthetic SkillMiner-style QA dataset
    dog_cat_qa.csv                 # dog–cat QA dataset from Kaggle 
```



### 4.2 Synthetic SkillMiner QA

- ~200 question–answer pairs
- Questions are designed so that:
  - **Short‑term memory (STM)** queries look back about **2 rows**
  - **Long‑term memory (LTM)** queries look back about **9 rows**
- Used as the main benchmark for ablation across Model 0–4.

### 4.3 Dog–Cat QA (real dataset, optional)

- Real QA pairs about dogs and cats (Kaggle dataset)
- Used to test cross‑domain generalization
- Place the CSV under `data/dog_cat_qa.csv` and run `train_qa_dog_cat.py`.

---

## 🧠 5. Model Variants

All models share `base_lstm_model.py` and differ only in how they use memory.

1. **Model 0 – Base LSTM** (`model_0_base.py`)  
   - Vanilla LSTM, no external memory.

2. **Model 1 – Summarization** (`model_1_summarization.py`)  
   - Adds a summarization layer on recent context.

3. **Model 2 – Summarization + Token Limit** (`model_2_sum_toklimit.py`)  
   - Enforces a max token budget so context stays bounded.

4. **Model 3 – Summarization + Token Limit + NER** (`model_3_sum_tok_ner.py`)  
   - Runs spaCy NER and stores entities in an auxiliary memory.

5. **Model 4 – Summarization + Token Limit + NER + Semantic** (`model_4_full_memory.py`)  
   - Combines summarization, token limit, NER and long‑term semantic retrieval.

---

## 🏋️ 6. Training & Running Experiments

### 6.1 Train on synthetic SkillMiner QA

```bash
python train_qa_memory.py   --model full_memory   --epochs 10 
```

Example `--model` choices depend on how you implemented the script, but typically:

```text
base, summarization, sum_toklimit, sum_tok_ner, full_memory
```

Check `train_qa_memory.py` for the exact argument names.

- Checkpoints are saved under `checkpoints_qa/`
- Logs are written into `logs/` and `cache_qa/` if caching is enabled

### 6.2 Train on dog–cat QA

```bash
python train_qa_dog_cat.py   --model full_memory   --epochs 10 
```

Checkpoints will be stored in `checkpoints_qa_dog_cat/` and caches in `cache_qa_dog_cat/`.

---

## 🧪 7. Evaluation

The training scripts typically perform evaluation at the end of each epoch and print summaries such as:

- Average loss  
- STM accuracy (LLM/difflib)  
- LTM accuracy (LLM/difflib)  

Typical evaluation rules:

- **STM** looks back ~2 previous QA rows  
- **LTM** looks back ~9 rows  
- LLM scores are thresholded (e.g. 0.6 for STM, 0.5 for LTM) to decide correctness.

If you only want offline evaluation, you can log predictions and post‑process them in  `nlp_final_project/`.

---

## 📈 8. Summary of Results

High‑level findings from this project:

- Adding **summarization + token limit** improves robustness over the base LSTM.  
- Adding **NER and long‑term semantic memory** further improves long‑range QA accuracy.  
- On synthetic data, **Model 4 (Full Memory)** achieves the best STM accuracy, while **Model 3 (NER)** reaches the best LTM accuracy.  
- On the real dog–cat dataset, models still reduce loss but accuracy is more sensitive to domain mismatch.

Readers can reproduce these trends by running `train_qa_memory.py` and checking logs in `logs/` or the report in `nlp_final_project/`.

---

## 🔗 9. Relationship to SkillMiner

This research repo is **not** the production SkillMiner codebase, but it directly inspired SkillMiner’s memory strategy.

This repo therefore serves as the **research companion** to the SkillMiner product.

---

## 📜 10. License

This project is released under the **MIT License**.  
Feel free to reuse or adapt the code with attribution.
