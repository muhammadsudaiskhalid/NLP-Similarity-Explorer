# NLP Similarity Explorer 🔍

**Shifa Tameer-e-Millat University — NLP Lab Quiz**

A Streamlit web app that uses the free pretrained model **all-MiniLM-L6-v2** (Sentence Transformers) to compute and visualise text/sentence similarity — with no preprocessing, no model training, and no paid APIs.

---

## 🚀 App Purpose

Enter a query sentence and a set of comparison sentences. The app:
- Encodes all sentences using a pretrained transformer
- Computes cosine similarity between the query and each comparison
- Displays results with **3 graphs** and **Paul's Critical Thinking Standards**

---

## 🤖 Pretrained Model

| Detail | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Source | [Hugging Face / Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Cost | Free — no API key required |
| Preprocessing | None — raw text is fed directly |

---

## 📊 Graphs Included

1. **Bar Chart** — top similar sentences ranked by cosine score  
2. **Heatmap** — pairwise similarity matrix for all sentences  
3. **PCA 2-D Plot** — sentence embeddings projected to 2-D space  

---

## 🧠 Paul's Critical Thinking Standards Applied

Clarity · Accuracy · Precision · Relevance · Logic · Significance · Fairness

---

## ⚙️ Run Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/nlp-similarity-app.git
cd nlp-similarity-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## 📁 Repository Structure

```
nlp-similarity-app/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🌐 Deployed App

> _(Add your Streamlit Community Cloud link here after deployment)_

---

## 📸 Screenshots

> _(Add screenshots of the running app here)_
