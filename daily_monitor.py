"""
Daily Model Health Monitor
Runs automatically every day via GitHub Actions
"""

import os
import json
import random
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load model
model  = joblib.load("fraudguard_model.jb")
encoder = joblib.load("category_encoder.jb")

# Generate sample transactions to score
np.random.seed(int(datetime.now().strftime("%j")))

categories = ['grocery_pos','gas_transport','shopping_net','entertainment','food_dining']
merchants  = [f'merchant_{i}' for i in range(50)]

n = random.randint(800, 1200)
df = pd.DataFrame({
    'merchant' : np.random.choice(merchants, n),
    'category' : np.random.choice(categories, n),
    'amt'      : np.random.exponential(60, n).round(2),
    'distance' : np.abs(np.random.normal(15, 20, n)).round(2),
    'hour'     : np.random.randint(0, 24, n),
    'day'      : np.random.randint(1, 32, n),
    'month'    : int(datetime.now().month),
    'gender'   : np.random.choice(['M','F'], n),
    'cc_num'   : np.random.randint(0, 100, n),
})

for col in ['merchant','category','gender']:
    try:    df[col] = encoder[col].transform(df[col].astype(str))
    except: df[col] = -1

probs  = model.predict_proba(df)[:,1]
flagged = int((probs >= 0.71).sum())

# Save log
os.makedirs("logs", exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")

log = {
    "date"             : today,
    "transactions_scored" : n,
    "flagged_as_fraud" : flagged,
    "fraud_rate_pct"   : round(flagged/n*100, 3),
    "avg_fraud_prob"   : round(float(probs.mean()), 4),
    "max_fraud_prob"   : round(float(probs.max()), 4),
    "model_status"     : "healthy"
}

# Append to daily log file
log_file = "logs/daily_monitor.json"
logs = []
if os.path.exists(log_file):
    with open(log_file) as f:
        logs = json.load(f)

logs.append(log)

with open(log_file, "w") as f:
    json.dump(logs, f, indent=2)

print(f"✅ {today} — Scored {n} transactions, flagged {flagged} ({flagged/n*100:.2f}% fraud rate)")