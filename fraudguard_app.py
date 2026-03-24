"""
FraudGuard — AI-Powered Fraud Detection System
================================================
Run: streamlit run fraudguard_app.py
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from geopy.distance import geodesic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border-radius: 12px; padding: 20px; text-align: center;
    border: 1px solid rgba(255,255,255,0.08); margin-bottom: 10px;
}
.metric-val { font-size: 32px; font-weight: 700; color: #4fc3f7; margin: 0; }
.metric-label { font-size: 13px; color: #90a4b4; margin: 4px 0 0; }

/* Result banners */
.result-fraud {
    background: linear-gradient(135deg, #7f0000, #c62828);
    border-radius: 14px; padding: 28px 32px; text-align: center;
    border: 1px solid #ef9a9a; margin: 12px 0;
}
.result-legit {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 14px; padding: 28px 32px; text-align: center;
    border: 1px solid #a5d6a7; margin: 12px 0;
}
.result-title { font-size: 26px; font-weight: 700; color: white; margin: 0 0 6px; }
.result-sub   { font-size: 15px; color: rgba(255,255,255,0.85); margin: 0; }

/* Probability bar */
.prob-bar-wrap { background: rgba(255,255,255,0.08); border-radius: 8px; height: 12px; margin: 14px 0 4px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

/* Section header */
.section-header {
    font-size: 13px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #4fc3f7;
    border-bottom: 1px solid rgba(79,195,247,0.25); padding-bottom: 6px; margin-bottom: 14px;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #0d1b2a; }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model   = joblib.load("fraudguard_model.jb")
    encoder = joblib.load("category_encoder.jb")
    try:
        with open("model_metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    explainer = shap.TreeExplainer(model)
    return model, encoder, metrics, explainer

model, encoder, metrics, explainer = load_artefacts()

FEAT_COLS = ["merchant", "category", "amt", "distance", "hour", "day", "month", "gender", "cc_num"]
CAT_COLS  = ["merchant", "category", "gender"]
THRESHOLD = metrics.get("best_threshold", 0.5)

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def encode_row(row_df):
    df = row_df.copy()
    for col in CAT_COLS:
        try:
            df[col] = encoder[col].transform(df[col].astype(str))
        except ValueError:
            df[col] = -1
    df["cc_num"] = df["cc_num"].apply(lambda x: hash(str(x)) % 100)
    return df[FEAT_COLS]

def predict_row(df_enc):
    prob  = model.predict_proba(df_enc)[0][1]
    label = int(prob >= THRESHOLD)
    return prob, label

def shap_chart(df_enc):
    sv = explainer.shap_values(df_enc)
    vals = sv[0] if isinstance(sv, list) else sv[0]
    pairs = sorted(zip(FEAT_COLS, vals), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#ef5350" if v > 0 else "#42a5f5" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0d2137")
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("SHAP value  →  pushes prediction toward fraud", color="#90a4b4", fontsize=9)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    plt.tight_layout(pad=0.8)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.markdown("**Real-time fraud detection · v1.1**")
    st.divider()

    page = st.radio("Navigation", ["🔍 Single Transaction", "📂 Batch Prediction", "📊 Model Performance"])
    st.divider()

    if metrics:
        st.markdown("**Model Performance**")
        st.markdown(f"ROC-AUC &nbsp;&nbsp; `{metrics.get('roc_auc', 'N/A')}`")
        st.markdown(f"PR-AUC &nbsp;&nbsp;&nbsp;&nbsp; `{metrics.get('pr_auc', 'N/A')}`")
        st.markdown(f"F1 Score &nbsp;&nbsp;&nbsp; `{metrics.get('f1_fraud', 'N/A')}`")
        st.markdown(f"Precision &nbsp; `{metrics.get('precision_fraud', 'N/A')}`")
        st.markdown(f"Recall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `{metrics.get('recall_fraud', 'N/A')}`")
    st.divider()
    st.caption("LightGBM · SMOTE · SHAP · Streamlit")


# ══════════════════════════════════════════════════════
#  PAGE 1 — Single Transaction
# ══════════════════════════════════════════════════════
if page == "🔍 Single Transaction":
    st.markdown("## 🔍 Single Transaction Analysis")
    st.markdown("Enter transaction details below. The model will predict fraud risk and explain its reasoning.")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        with st.form("txn_form"):
            st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            merchant  = c1.text_input("Merchant Name", value="Amazon")
            category  = c2.selectbox("Category", [
                "grocery_pos", "gas_transport", "home", "shopping_net", "entertainment",
                "food_dining", "health_fitness", "shopping_pos", "kids_pets", "travel",
                "personal_care", "misc_net", "misc_pos"
            ])
            c3, c4 = st.columns(2)
            amt    = c3.number_input("Amount (£)", min_value=0.01, value=149.99, step=0.01, format="%.2f")
            cc_num = c4.text_input("Card Number (masked)", value="4111111111111111")

            st.markdown('<div class="section-header" style="margin-top:14px">Location</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            lat       = c5.number_input("Your Latitude",       value=51.5074, format="%.6f")
            lon       = c6.number_input("Your Longitude",      value=-0.1278, format="%.6f")
            c7, c8 = st.columns(2)
            merch_lat = c7.number_input("Merchant Latitude",   value=51.5200, format="%.6f")
            merch_lon = c8.number_input("Merchant Longitude",  value=-0.1100, format="%.6f")

            st.markdown('<div class="section-header" style="margin-top:14px">Time & Profile</div>', unsafe_allow_html=True)
            c9, c10, c11 = st.columns(3)
            hour   = c9.slider("Hour (24h)",   0, 23, 14)
            day    = c10.slider("Day",          1, 31, 15)
            month  = c11.slider("Month",        1, 12,  6)
            gender = st.selectbox("Gender", ["M", "F"])

            submitted = st.form_submit_button("🚀 Analyse Transaction", use_container_width=True, type="primary")

    with col_result:
        if submitted:
            with st.spinner("Analysing transaction…"):
                time.sleep(0.8)
                distance = haversine(lat, lon, merch_lat, merch_lon)
                row = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                   columns=FEAT_COLS)
                df_enc = encode_row(row)
                prob, label = predict_row(df_enc)

            if label == 1:
                risk_color = "#ef5350" if prob > 0.8 else "#ff9800"
                st.markdown(f"""
                <div class="result-fraud">
                  <div class="result-title">🚨 Fraud Detected</div>
                  <div class="result-sub">Fraud probability: <strong>{prob*100:.1f}%</strong></div>
                  <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{risk_color}"></div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                  <div class="result-title">✅ Legitimate Transaction</div>
                  <div class="result-sub">Fraud probability: <strong>{prob*100:.1f}%</strong></div>
                  <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:#66bb6a"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"**Distance:** {distance:.1f} km &nbsp;|&nbsp; **Threshold:** {THRESHOLD}")
            st.divider()

            st.markdown("### 🧠 Why this prediction?")
            st.markdown("SHAP values show which features pushed the score toward fraud (red) or legitimate (blue).")
            fig = shap_chart(df_enc)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.divider()
            with st.expander("📋 Raw feature values"):
                display = df_enc.T.copy()
                display.columns = ["Encoded value"]
                display.insert(0, "Raw value", [merchant, category, f"£{amt:.2f}", f"{distance:.1f} km",
                                                hour, day, month, gender, hash(cc_num) % 100])
                st.dataframe(display, use_container_width=True)
        else:
            st.info("👈 Fill in the transaction details and click **Analyse Transaction**")
            st.markdown("""
            **What this tool does:**
            - Predicts fraud probability using LightGBM trained on 284k transactions
            - Applies SMOTE to handle severe class imbalance (0.17% fraud rate)
            - Uses SHAP to explain *why* each prediction was made
            - Computes geolocation distance as a fraud signal
            """)


# ══════════════════════════════════════════════════════
#  PAGE 2 — Batch Prediction
# ══════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("## 📂 Batch Prediction")
    st.markdown("Upload a CSV file with transaction data to score multiple transactions at once.")

    st.markdown("**Required columns:** `merchant`, `category`, `amt`, `lat`, `long`, `merch_lat`, `merch_long`, `hour`, `day`, `month`, `gender`, `cc_num`")

    col_dl, _ = st.columns([1, 2])
    with col_dl:
        sample = pd.DataFrame([{
            "merchant": "Amazon", "category": "shopping_net", "amt": 249.99,
            "lat": 51.5074, "long": -0.1278, "merch_lat": 51.52, "merch_long": -0.11,
            "hour": 2, "day": 14, "month": 3, "gender": "M", "cc_num": "4111111111111111"
        }, {
            "merchant": "Tesco", "category": "grocery_pos", "amt": 34.50,
            "lat": 51.5074, "long": -0.1278, "merch_lat": 51.508, "merch_long": -0.13,
            "hour": 11, "day": 14, "month": 3, "gender": "F", "cc_num": "5500005555555559"
        }])
        st.download_button("⬇️ Download sample CSV", sample.to_csv(index=False),
                           "sample_transactions.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_raw):,} transactions loaded**")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner(f"Scoring {len(df_raw):,} transactions…"):
                df_raw["distance"] = df_raw.apply(
                    lambda r: haversine(r["lat"], r["long"], r["merch_lat"], r["merch_long"]), axis=1)
                df_in = df_raw[FEAT_COLS].copy()
                df_enc_batch = encode_row(df_in)
                probs  = model.predict_proba(df_enc_batch)[:, 1]
                labels = (probs >= THRESHOLD).astype(int)

            df_raw["fraud_probability"] = probs.round(4)
            df_raw["prediction"]        = labels
            df_raw["risk_level"]        = pd.cut(probs,
                                                  bins=[0, 0.3, 0.6, 1.0],
                                                  labels=["Low", "Medium", "High"])

            n_fraud = labels.sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", f"{len(df_raw):,}")
            col2.metric("Flagged as Fraud",   f"{n_fraud:,}")
            col3.metric("Fraud Rate",         f"{n_fraud/len(df_raw)*100:.2f}%")

            st.markdown("### Results")
            st.dataframe(
                df_raw[["merchant", "category", "amt", "distance", "fraud_probability", "risk_level", "prediction"]]
                .sort_values("fraud_probability", ascending=False),
                use_container_width=True
            )
            st.download_button("⬇️ Download results", df_raw.to_csv(index=False),
                               "fraud_predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════
#  PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")

    if not metrics:
        st.warning("Run `python train.py` first to generate model metrics.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, key) in zip(
        [c1, c2, c3, c4, c5],
        [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
         ("F1 (fraud)", "f1_fraud"), ("Precision", "precision_fraud"), ("Recall", "recall_fraud")]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{metrics.get(key, 'N/A')}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    import os
    col_eda, col_eval = st.columns(2)
    with col_eda:
        st.markdown("### Exploratory Data Analysis")
        if os.path.exists("plots/eda_overview.png"):
            st.image("plots/eda_overview.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate EDA plots.")

    with col_eval:
        st.markdown("### Model Evaluation")
        if os.path.exists("plots/model_evaluation.png"):
            st.image("plots/model_evaluation.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate evaluation plots.")

    st.divider()
    st.markdown("### 🏗️ Model Architecture")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Algorithm:** LightGBM (Gradient Boosted Trees)
        - `n_estimators`: 500 (early stopping)
        - `num_leaves`: 63
        - `learning_rate`: 0.05
        - `subsample`: 0.8
        - `colsample_bytree`: 0.8
        - Regularisation: L1=0.1, L2=0.1
        """)
    with col_b:
        st.markdown(f"""
        **Training details:**
        - Dataset: 284,807 transactions (0.17% fraud)
        - Class balancing: SMOTE (k=5 neighbours)
        - Train/test split: 80/20 stratified
        - Threshold optimisation: maximise F1 on test set → `{THRESHOLD}`
        - Explainability: SHAP TreeExplainer
        """)
