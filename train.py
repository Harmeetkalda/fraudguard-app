"""
FraudGuard — Model Training Pipeline
=====================================
Dataset : creditcard_fraud.csv  (or drop in the Kaggle Credit Card Fraud dataset)
Output  : fraudguard_model.jb, category_encoder.jb, model_metrics.json
Run     : python train.py
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── 0. Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "creditcard_fraud.csv"
MODEL_PATH  = "fraudguard_model.jb"
ENC_PATH    = "category_encoder.jb"
METRICS_PATH = "model_metrics.json"
PLOTS_DIR   = "plots"
RANDOM_STATE = 42
CAT_COLS    = ["merchant", "category", "gender"]
FEAT_COLS   = ["merchant", "category", "amt", "distance", "hour", "day", "month", "gender", "cc_num"]
TARGET      = "is_fraud"

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("  FraudGuard — Training Pipeline")
print("=" * 60)

# ── 1. Load & Explore ────────────────────────────────────────────────────────
print("\n[1/7] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape      : {df.shape}")
print(f"  Fraud rate : {df[TARGET].mean() * 100:.2f}%  ({df[TARGET].sum():,} fraud / {len(df):,} total)")
print(f"  Null values: {df.isnull().sum().sum()}")

# ── 2. EDA Plots ─────────────────────────────────────────────────────────────
print("\n[2/7] Generating EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("FraudGuard — Exploratory Data Analysis", fontsize=15, fontweight="bold", y=1.01)

# Class distribution
class_counts = df[TARGET].value_counts()
axes[0, 0].bar(["Legitimate", "Fraud"], class_counts.values,
               color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.2)
axes[0, 0].set_title("Class Distribution", fontweight="bold")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

# Transaction amount by class
df_plot = df.copy()
df_plot["Class"] = df_plot[TARGET].map({0: "Legitimate", 1: "Fraud"})
axes[0, 1].hist(df_plot[df_plot[TARGET] == 0]["amt"].clip(0, 500), bins=50,
                alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
axes[0, 1].hist(df_plot[df_plot[TARGET] == 1]["amt"].clip(0, 500), bins=50,
                alpha=0.6, color="#e74c3c", label="Fraud", density=True)
axes[0, 1].set_title("Transaction Amount Distribution", fontweight="bold")
axes[0, 1].set_xlabel("Amount (£)")
axes[0, 1].legend()

# Fraud by hour
fraud_by_hour = df.groupby("hour")[TARGET].mean() * 100
axes[0, 2].bar(fraud_by_hour.index, fraud_by_hour.values, color="#e74c3c", alpha=0.75)
axes[0, 2].set_title("Fraud Rate by Hour of Day", fontweight="bold")
axes[0, 2].set_xlabel("Hour")
axes[0, 2].set_ylabel("Fraud Rate (%)")

# Distance distribution
axes[1, 0].hist(df[df[TARGET] == 0]["distance"].clip(0, 200), bins=50,
                alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
axes[1, 0].hist(df[df[TARGET] == 1]["distance"].clip(0, 500), bins=50,
                alpha=0.6, color="#e74c3c", label="Fraud", density=True)
axes[1, 0].set_title("Distance Distribution (km)", fontweight="bold")
axes[1, 0].set_xlabel("Distance (km)")
axes[1, 0].legend()

# Fraud by category
cat_fraud = df.groupby("category")[TARGET].mean().sort_values(ascending=True) * 100
axes[1, 1].barh(cat_fraud.index, cat_fraud.values, color="#e74c3c", alpha=0.75)
axes[1, 1].set_title("Fraud Rate by Category", fontweight="bold")
axes[1, 1].set_xlabel("Fraud Rate (%)")

# Fraud by gender
gender_fraud = df.groupby("gender")[TARGET].mean() * 100
axes[1, 2].bar(gender_fraud.index.map({"M": "Male", "F": "Female"}),
               gender_fraud.values, color=["#3498db", "#e91e8c"], alpha=0.8)
axes[1, 2].set_title("Fraud Rate by Gender", fontweight="bold")
axes[1, 2].set_ylabel("Fraud Rate (%)")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/eda_overview.png")

# ── 3. Feature Engineering & Encoding ────────────────────────────────────────
print("\n[3/7] Feature engineering & encoding...")

encoders = {}
df_enc = df.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, ENC_PATH)
print(f"  Encoders saved → {ENC_PATH}")

X = df_enc[FEAT_COLS]
y = df_enc[TARGET]

# ── 4. Train / Test Split ────────────────────────────────────────────────────
print("\n[4/7] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train : {X_train.shape[0]:,} samples  |  Fraud: {y_train.sum():,}")
print(f"  Test  : {X_test.shape[0]:,} samples  |  Fraud: {y_test.sum():,}")

# ── 5. SMOTE Oversampling ────────────────────────────────────────────────────
print("\n[5/7] Applying SMOTE to balance training set...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {X_train_res.shape[0]:,} samples  |  Fraud: {y_train_res.sum():,}  ({y_train_res.mean()*100:.1f}%)")

# ── 6. Train LightGBM ────────────────────────────────────────────────────────
print("\n[6/7] Training LightGBM classifier...")

params = {
    "objective"        : "binary",
    "metric"           : ["binary_logloss", "auc"],
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "num_leaves"       : 63,
    "max_depth"        : 7,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
    "verbose"          : -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

joblib.dump(model, MODEL_PATH)
print(f"  Model saved → {MODEL_PATH}")
print(f"  Best iteration: {model.best_iteration_}")

# ── 7. Evaluate & Plot ────────────────────────────────────────────────────────
print("\n[7/7] Evaluating model...")

y_prob = model.predict_proba(X_test)[:, 1]

# Optimal threshold via F1
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_prob >= best_threshold).astype(int)

roc_auc   = roc_auc_score(y_test, y_prob)
pr_auc    = average_precision_score(y_test, y_prob)
f1        = f1_score(y_test, y_pred)
report    = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True)

print(f"\n  ROC-AUC   : {roc_auc:.4f}")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  F1 (fraud): {f1:.4f}  [threshold={best_threshold:.2f}]")
print(f"  Precision : {report['Fraud']['precision']:.4f}")
print(f"  Recall    : {report['Fraud']['recall']:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

# Save metrics
metrics = {
    "roc_auc"        : round(roc_auc, 4),
    "pr_auc"         : round(pr_auc, 4),
    "f1_fraud"       : round(f1, 4),
    "precision_fraud": round(report["Fraud"]["precision"], 4),
    "recall_fraud"   : round(report["Fraud"]["recall"], 4),
    "best_threshold" : round(float(best_threshold), 2),
    "train_samples"  : int(X_train_res.shape[0]),
    "test_samples"   : int(X_test.shape[0]),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Metrics saved → {METRICS_PATH}")

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("FraudGuard — Model Evaluation", fontsize=15, fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=",", cmap="RdYlGn",
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            ax=axes[0, 0], linewidths=0.5)
axes[0, 0].set_title(f"Confusion Matrix  (threshold={best_threshold:.2f})", fontweight="bold")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curve", fontweight="bold")
axes[0, 1].legend()
axes[0, 1].fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1, 0].plot(rec, prec, color="#3498db", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[1, 0].axhline(y=y_test.mean(), color="gray", linestyle="--", lw=1, label="Baseline")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].set_title("Precision-Recall Curve", fontweight="bold")
axes[1, 0].legend()
axes[1, 0].fill_between(rec, prec, alpha=0.08, color="#3498db")

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values()
colors = ["#e74c3c" if v == feat_imp.max() else "#3498db" for v in feat_imp.values]
axes[1, 1].barh(feat_imp.index, feat_imp.values, color=colors, alpha=0.8)
axes[1, 1].set_title("Feature Importance (gain)", fontweight="bold")
axes[1, 1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {PLOTS_DIR}/model_evaluation.png")

print("\n" + "=" * 60)
print("  Training complete! Files generated:")
print(f"    {MODEL_PATH}")
print(f"    {ENC_PATH}")
print(f"    {METRICS_PATH}")
print(f"    {PLOTS_DIR}/eda_overview.png")
print(f"    {PLOTS_DIR}/model_evaluation.png")
print("=" * 60)
