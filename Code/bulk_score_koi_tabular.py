# =========================================================
# Space Invaders — Bulk Score ~500 KOIs using Tabular Model
# (TAP fixed + classic API fallback + model feature auto-align)
# =========================================================

import os, json, argparse, io
import requests
import pandas as pd
import numpy as np
import joblib

# ---------- Feature sets ----------
FULL_FEATURES_15 = [
    "koi_period","koi_duration","koi_depth","koi_impact","koi_prad",
    "koi_teq","koi_insol","koi_model_snr","koi_steff","koi_slogg",
    "koi_srad","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"
]
NUMERIC_FEATURES_11 = [
    # same order as first 11 of FULL_FEATURES_15 (without flags)
    "koi_period","koi_duration","koi_depth","koi_impact","koi_prad",
    "koi_teq","koi_insol","koi_model_snr","koi_steff","koi_slogg","koi_srad"
]
ID_COLS = ["kepid","kepoi_name","kepler_name","koi_disposition"]

# ---------- TAP helpers ----------
TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CLASSIC_API = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
ADQL_SELECT = (
    "SELECT kepid, kepoi_name, kepler_name, koi_disposition, "
    "koi_period, koi_duration, koi_depth, koi_impact, koi_prad, koi_teq, koi_insol, "
    "koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec "
)

def _tap_query(adql):
    params = {"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"csv","QUERY":adql}
    r = requests.get(TAP_SYNC, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_koi_csv():
    # Try FROM koi
    try:
        text = _tap_query(ADQL_SELECT + "FROM koi")
        return pd.read_csv(io.StringIO(text))
    except Exception as e1:
        # Try FROM exoplanetarchive:koi
        try:
            text = _tap_query(ADQL_SELECT + "FROM exoplanetarchive:koi")
            return pd.read_csv(io.StringIO(text))
        except Exception as e2:
            # Classic API fallback
            params = {
                "table":"koi",
                "select":",".join(ID_COLS + FULL_FEATURES_15),
                "format":"csv"
            }
            r = requests.get(CLASSIC_API, params=params, timeout=60)
            try:
                r.raise_for_status()
                return pd.read_csv(io.StringIO(r.text))
            except Exception as e3:
                raise RuntimeError(
                    f"TAP and classic API both failed.\nTAP errors: {e1} | {e2}\nClassic error: {e3}"
                )

# ---------- Metadata helpers ----------
def load_feature_order_from_metadata(model_path: str):
    # Look next to the Combined directory (…/Processed/Combined/feature_metadata.json)
    proc_dir = os.path.dirname(os.path.dirname(model_path))
    meta_path = os.path.join(proc_dir, "feature_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path,"r",encoding="utf-8") as f:
                meta = json.load(f)
            feats = meta.get("feature_columns") or meta.get("features")
            if feats and all(isinstance(x,str) for x in feats):
                return feats
        except Exception:
            pass
    return None

def get_model_expected_n_and_names(model):
    """Return (n_features_expected, feature_names_or_None) from various model types."""
    # sklearn-style
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return len(names), list(names)

    # xgboost.XGBClassifier / Booster
    try:
        import xgboost as xgb  # noqa
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            n = booster.num_features()
            names = booster.feature_names
            return n, list(names) if names else None
        if model.__class__.__name__ == "Booster":
            n = model.num_features()
            names = model.feature_names
            return n, list(names) if names else None
        # our unpickled wrapper may store .booster
        if hasattr(model, "booster"):
            n = model.booster.num_features()
            names = model.booster.feature_names
            return n, list(names) if names else None
    except Exception:
        pass

    # no info
    return None, None

# ---------- Cleaning ----------
def coerce_and_impute(df, feature_cols):
    # numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # fill flags with 0 if flags present
    for c in ["koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("Int64")

    # median for continuous
    for c in feature_cols:
        if c not in ["koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"]:
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)

    # final float cast
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    return df

# ---------- Label mapping ----------
def map_labels_from_classes_attr(class_indices, classes_attr):
    # classes_ may be e.g. [-1,0,1] (FP, CAND, CONF)
    def name_for(val):
        if val <= -1: return "FALSE POSITIVE"
        if val == 0:  return "CANDIDATE"
        return "CONFIRMED"
    idx_to_val = {i:int(v) for i,v in enumerate(list(classes_attr))}
    names = []
    for idx in class_indices:
        v = idx_to_val.get(int(idx), idx)
        names.append(name_for(v))
    return names

# ---------- Unpickle shim for wrapper ----------
try:
    import xgboost as xgb
except Exception:
    xgb = None

class XGBBoosterWrapper:  # name must match the pickled class
    def predict_proba(self, X):
        if xgb is None:
            raise RuntimeError("xgboost not installed.")
        dm = xgb.DMatrix(X)
        return self.booster.predict(dm)
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to your Combined tabular model.pkl")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--n", type=int, default=500, help="How many rows to score (after cleaning)")
    args = ap.parse_args()

    print("⏬ Downloading KOI table from NASA Exoplanet Archive …")
    df = fetch_koi_csv()
    print(f"Total rows fetched: {len(df)}")

    # Keep only needed columns (we’ll choose which feature set later)
    base_cols = set(ID_COLS + FULL_FEATURES_15)
    df = df[[c for c in df.columns if c in base_cols]].copy()

    # Load model (shim class already defined above)
    model_obj = joblib.load(args.model)
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        classes_attr = model_obj.get("classes_", None) or model_obj.get("classes", None)
    else:
        model = model_obj
        classes_attr = getattr(model, "classes_", None)

    # Determine expected features
    feat_from_meta = load_feature_order_from_metadata(args.model)
    expected_n, model_names = get_model_expected_n_and_names(model)

    # Choose feature order
    if feat_from_meta:
        feat_order = [f for f in feat_from_meta if f in df.columns]
        # If metadata exists but doesn’t match expected_n, we still trust metadata order.
    elif expected_n == 11:
        feat_order = [f for f in NUMERIC_FEATURES_11 if f in df.columns]
    elif expected_n == 15:
        feat_order = [f for f in FULL_FEATURES_15 if f in df.columns]
    else:
        # Fallback: prefer numeric 11 if all present, else use full 15 then trim
        if all(f in df.columns for f in NUMERIC_FEATURES_11):
            feat_order = NUMERIC_FEATURES_11[:]
        else:
            feat_order = [f for f in FULL_FEATURES_15 if f in df.columns]

    # Final trim/align to expected_n if we know it
    if expected_n is not None:
        if len(feat_order) > expected_n:
            feat_order = feat_order[:expected_n]
        elif len(feat_order) < expected_n:
            # try to append remaining from preferred lists
            pool = [c for c in (FULL_FEATURES_15 if expected_n > 11 else NUMERIC_FEATURES_11) if c not in feat_order and c in df.columns]
            need = expected_n - len(feat_order)
            feat_order += pool[:need]

    # Validate we have something
    if len(feat_order) == 0:
        raise RuntimeError("Could not determine a usable feature order for this model.")

    print(f"🔧 Using {len(feat_order)} features for scoring:")
    print("   " + ", ".join(feat_order))

    # Clean & impute only those features
    df_clean = coerce_and_impute(df.copy(), feat_order)

    # Keep valid rows and limit to N
    mask_valid = df_clean[feat_order].notna().all(axis=1)
    df_valid = df_clean.loc[mask_valid].head(args.n).reset_index(drop=True)
    print(f"Rows to score (valid & limited to n={args.n}): {len(df_valid)}")

    # Predict
    X = df_valid[feat_order].values
    # Try proba
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    yhat = model.predict(X)

    # Human-readable labels
    if classes_attr is not None:
        label_names = map_labels_from_classes_attr(yhat, classes_attr)
    else:
        # Fallback mapping
        label_names = []
        for v in yhat:
            if int(v) <= -1: label_names.append("FALSE POSITIVE")
            elif int(v) == 0: label_names.append("CANDIDATE")
            else: label_names.append("CONFIRMED")

    # Build clean output
    out = df_valid[ID_COLS].copy()
    out["Pred_Class_Index"] = yhat
    out["Pred_Label"] = label_names
    if proba is not None and proba.ndim == 2:
        for j in range(proba.shape[1]):
            out[f"Prob_class_{j}"] = proba[:, j]

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✅ Saved predictions → {args.out}")

    # Preview
    preview_cols = ID_COLS + ["Pred_Label"]
    print("\nFirst few rows:")
    print(out[preview_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
