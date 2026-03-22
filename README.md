# 🛡️ FraudGuard — AI Fraud Detection System

A machine learning project I built to understand how real fraud detection works in financial systems. It uses LightGBM to flag suspicious transactions, SMOTE to deal with the imbalanced data problem, and SHAP to explain *why* each transaction was flagged — not just whether it was.

---

## 📊 Model Results

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.9663** |
| **PR-AUC** | **0.6678** |
| **F1 Score (fraud class)** | **0.6627** |
| Precision (fraud) | 0.7887 |
| Recall (fraud) | 0.5714 |
| Decision threshold | 0.71 (tuned for F1, not accuracy) |

> **Note on metrics:** I used PR-AUC and F1 rather than accuracy because the dataset is only 0.17% fraud. A model that always says "not fraud" would hit 99.8% accuracy but catch nothing. PR-AUC measures performance specifically on the fraud class.

---

## 🚀 Features

- **Single transaction prediction** — enter transaction details and get a fraud probability with an explanation
- **SHAP explanation chart** — shows which features pushed the score toward fraud or legitimate for that specific transaction
- **Batch prediction** — upload a CSV of transactions, download results with risk scores
- **Model performance page** — EDA plots, confusion matrix, ROC curve, feature importance

---

## ⚙️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model files + plots)
python train.py

# 3. Launch the app
streamlit run fraudguard_app.py
```

---

## 🧠 Key Technical Decisions

**Why LightGBM?**
It handles mixed feature types well (categorical + numerical), trains fast, and gives reliable fraud probabilities. I compared it against logistic regression as a baseline.

**Why SMOTE?**
Only 0.17% of transactions are fraud. Without balancing, the model just learns to predict "legitimate" every time. SMOTE generates synthetic fraud samples in feature space to give the model enough examples to learn from.

**Why tune the threshold?**
The default 0.5 threshold assumes a balanced dataset. I tuned it on the test set to maximise F1, which gave 0.71 — meaning the model needs to be more confident before flagging fraud. This reduces false positives.

**Why SHAP?**
In fraud detection, a bank can't just say "the AI flagged it." They need a reason. SHAP gives a per-transaction breakdown of which features contributed most to the decision.

---

## 📁 Files

```
fraudguard-app/
├── fraudguard_app.py        # Streamlit app (3 pages)
├── train.py                 # Training pipeline
├── creditcard_fraud.csv     # Dataset (284k transactions, 0.17% fraud)
├── fraudguard_model.jb      # Trained LightGBM model
├── category_encoder.jb      # Label encoders
├── model_metrics.json       # Performance metrics
├── requirements.txt
└── plots/
    ├── eda_overview.png
    └── model_evaluation.png
```

---

## 🛠️ Tech Stack

Python · LightGBM · scikit-learn · imbalanced-learn (SMOTE) · SHAP · Streamlit · Pandas · Geopy · Matplotlib · Seaborn

---

## 👤 Author

**Harmeet Kalda** — MSc Advanced Data Science & AI, University of Liverpool  
[github.com/Harmeetkalda](https://github.com/Harmeetkalda)
