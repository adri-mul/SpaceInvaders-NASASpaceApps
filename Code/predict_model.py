# ============================================
# Space Invaders — Model Prediction Script (robust for XGBBoosterWrapper)
# ============================================

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# PATHS (fixed to your repo)
# ----------------------------
BASE_DIR = Path(r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps")
MODEL_PATH = BASE_DIR / "Processed" / "Combined" / "Models" / "XGB" / "model.pkl"
META_PATH  = BASE_DIR / "Processed" / "Combined" / "feature_metadata.json"
CSV_PATH   = BASE_DIR / "Data" / "new_exoplanet_data.csv"  # optional input

# ----------------------------
# Compatibility wrapper
# (must have the SAME name used during training)
# ----------------------------
class XGBBoosterWrapper:
    """Wrapper for xgboost.train Booster so it can be unpickled and used like sklearn."""
    def __init__(self, booster, num_class: int):
        self.booster = booster
        self.num_class = num_class

    def _as_dmatrix(self, X):
        import xgboost as xgb
        return xgb.DMatrix(X)

    def predict(self, X):
        import numpy as np
        dm = self._as_dmatrix(X)
        if self.num_class <= 2:
            prob = self.booster.predict(dm)  # prob of positive class
            return (prob >= 0.5).astype(int)
        else:
            prob = self.booster.predict(dm)  # (n, num_class)
            return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        dm = self._as_dmatrix(X)
        prob = self.booster.predict(dm)
        # Ensure 2D shape for binary
        if prob.ndim == 1:
            prob = np.vstack([1.0 - prob, prob]).T
        return prob

    def get_fscore_dict(self):
        try:
            return self.booster.get_fscore()
        except Exception:
            try:
                return self.booster.get_score(importance_type="weight")
            except Exception:
                return {}

# ----------------------------
# LOAD MODEL + METADATA
# ----------------------------
print(f"📦 Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)  # works for sklearn models or XGBBoosterWrapper

print(f"🧠 Loading metadata from {META_PATH}")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
feature_cols = meta.get("feature_cols", [])
label_map = meta.get("label_map")  # e.g., {"CANDIDATE": 0, "CONFIRMED": 1, "FALSE POSITIVE": 2}
inv_label_map = None
if label_map:
    inv_label_map = {int(v): k for k, v in label_map.items()}  # 0->"CANDIDATE", etc.

print("✅ Model and metadata loaded.\n")

# ----------------------------
# LOAD NEW DATA
# ----------------------------
if CSV_PATH.exists():
    print(f"🪐 Loading new dataset: {CSV_PATH}")
    new_data = pd.read_csv(CSV_PATH)
else:
    print("⚠️ No new_exoplanet_data.csv found. Using dummy row...\n"
          "   (Put a CSV at Data/new_exoplanet_data.csv to run real predictions)")
    # Dummy values with your feature schema—adjust if needed
    # If your feature_cols differ, we’ll auto-add missing columns below.
    new_data = pd.DataFrame([{
        "koi_period": 45.3,
        "koi_duration": 0.3,
        "koi_depth": 150,
        "koi_impact": 0.5,
        "koi_prad": 1.2,
        "koi_teq": 800,
        "koi_insol": 250,
        "koi_model_snr": 15.7,
        "koi_steff": 5400,
        "koi_slogg": 4.3,
        "koi_srad": 0.9,
        "koi_fpflag_nt": 1,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0
    }])

# Ensure all required columns exist, fill missing with 0
missing = [c for c in feature_cols if c not in new_data.columns]
for c in missing:
    new_data[c] = 0
# Keep only the required order
new_data = new_data[feature_cols]

# ----------------------------
# PREDICT
# ----------------------------
print(f"🔭 Predicting {len(new_data)} rows...")
has_proba = hasattr(model, "predict_proba")
if has_proba:
    probs = model.predict_proba(new_data.values if hasattr(new_data, "values") else new_data)
    # Convert to class ids
    preds = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
else:
    probs = None
    preds = model.predict(new_data.values if hasattr(new_data, "values") else new_data)

# Map predictions back to labels if we have a label_map
if inv_label_map:
    pred_labels = [inv_label_map.get(int(p), str(int(p))) for p in preds]
else:
    pred_labels = [str(int(p)) for p in preds]

# ----------------------------
# OUTPUT
# ----------------------------
out = new_data.copy()
out["Predicted_Label"] = pred_labels

if probs is not None:
    # If we know class names from label_map, add per-class columns
    if inv_label_map and isinstance(inv_label_map, dict) and probs.ndim == 2:
        # Build columns in the order of class index
        max_c = probs.shape[1]
        for cls_idx in range(max_c):
            cls_name = inv_label_map.get(cls_idx, f"class_{cls_idx}")
            safe_name = f"Prob_{cls_name.replace(' ', '_')}"
            out[safe_name] = probs[:, cls_idx]
    else:
        # generic positive prob for binary
        if probs.ndim == 2 and probs.shape[1] == 2:
            out["Prob_Positive"] = probs[:, 1]

print("\n=== Prediction Preview (first 10 rows) ===")
cols_to_show = [c for c in out.columns if c.startswith("Prob_")]
print(out[["Predicted_Label"] + cols_to_show].head(10))

OUT_PATH = BASE_DIR / "Processed" / "Combined" / "predictions.csv"
out.to_csv(OUT_PATH, index=False)
print(f"\n✅ Predictions saved -> {OUT_PATH}")
    