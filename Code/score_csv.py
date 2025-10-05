# =====================================================
# Space Invaders — Planet Predictor (Interactive Version)
# =====================================================

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import xgboost as xgb

# --- FIXED PATHS TO YOUR MODEL + METADATA ---
BASE_DIR = Path(r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps")
MODEL_PATH = BASE_DIR / "Processed" / "Combined" / "Models" / "XGB" / "model.pkl"
META_PATH  = BASE_DIR / "Processed" / "Combined" / "feature_metadata.json"

# --- Load the trained model ---
print("📦 Loading trained model...")
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")

# --- Load metadata to know what features to expect ---
print("🧠 Loading metadata...")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
feature_cols = meta.get("feature_cols", [])
label_map = meta.get("label_map", {"0": "CANDIDATE", "1": "CONFIRMED", "-1": "FALSE POSITIVE"})
inv_label_map = {int(v): k for k, v in label_map.items()} if all(v.isnumeric() for v in label_map.values()) else None
print(f"✅ Features expected: {len(feature_cols)} columns")

# --- Ask user for input CSV path ---
csv_path_str = input("\n📥 Enter full path to your CSV file of points: ").strip('"').strip()
csv_path = Path(csv_path_str)

if not csv_path.exists():
    print(f"❌ ERROR: File not found at {csv_path}")
    exit(1)

# --- Load the user's data ---
print(f"🪐 Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# --- Make sure required columns exist ---
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"⚠️ Missing columns in input — filling with 0: {missing}")
    for c in missing:
        df[c] = 0

# --- Keep only the expected feature order ---
X = df[feature_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# --- Predict using the trained model ---
print(f"🔭 Running model predictions on {len(X)} data points...")
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
else:
    probs = None
    preds = model.predict(X)

# --- Map predictions back to human-readable labels ---
def normalize_label(label):
    label = str(label).upper()
    if "CONFIRM" in label:
        return "CONFIRMED"
    if "FALSE" in label or "FP" in label or "REFUTE" in label:
        return "FALSE POSITIVE"
    if "CANDIDATE" in label or "PC" in label:
        return "CANDIDATE"
    return label

if inv_label_map:
    labels = [normalize_label(inv_label_map.get(int(p), str(p))) for p in preds]
else:
    labels = [normalize_label(str(p)) for p in preds]

# --- Compute simple verdicts ---
def verdict(lbl, p_conf, p_cand, p_fp):
    if lbl == "CONFIRMED" or p_conf >= 0.7:
        return "✅ Likely Confirmed"
    if lbl == "CANDIDATE" and p_cand >= 0.6:
        return "🟡 Strong Candidate"
    if p_fp >= 0.6:
        return "❌ Likely False Positive"
    return "🤔 Uncertain"

verdicts = []
if probs is not None and probs.ndim == 2 and probs.shape[1] >= 3:
    for i, lbl in enumerate(labels):
        p_conf, p_cand, p_fp = probs[i, 0], probs[i, 1], probs[i, 2]
        verdicts.append(verdict(lbl, p_conf, p_cand, p_fp))
else:
    verdicts = ["🤔 Uncertain"] * len(labels)

# --- Create output dataframe ---
out_df = df.copy()
out_df["Predicted_Label"] = labels
out_df["Verdict"] = verdicts
if probs is not None:
    if probs.shape[1] == 3:
        out_df["Prob_CONFIRMED"] = probs[:, 0]
        out_df["Prob_CANDIDATE"] = probs[:, 1]
        out_df["Prob_FALSE_POSITIVE"] = probs[:, 2]

# --- Save output ---
out_path = csv_path.with_name(csv_path.stem + "_predicted.csv")
out_df.to_csv(out_path, index=False)
print(f"\n✅ Results saved to: {out_path}")

# --- Summary of findings ---
summary = out_df["Predicted_Label"].value_counts().to_dict()
print("\n📊 Prediction Summary:")
for k, v in summary.items():
    print(f"   {k:>15}: {v}")

print("\n🌟 Done! You can now open the results CSV to see planet predictions.")
