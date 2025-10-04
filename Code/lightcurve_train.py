# ===============================================================
# Space Invaders — Train a Planet Classifier Directly from Light Curves
# - Downloads TESS/Kepler light curves by TIC ID (or reads local CSV with time/flux)
# - Extracts features via fast BLS (coarse→fine) with period/duration safety guard
# - Robust SNR calculation (works with Astropy versions w/o `res.snr`)
# - (Optional) fetches Teff/logg/radius from TIC catalog
# - Builds dataset, trains model, saves model.pkl + feature_metadata.json
# Usage examples (PowerShell):
#   python Code\lightcurve_train.py --tic_ids 150428135 307210830 123456789 --mission tess --labels_csv Data\labels.csv --model xgb --outdir Processed\LC_Model --min_period 1.0 --max_period 15 --coarse 800 --fine 250
#   python Code\lightcurve_train.py --csv_glob "Data\LightCurves\*.csv" --labels_csv Data\labels.csv --model xgb
#   python Code\lightcurve_train.py --tic_ids 150428135 --features_only
# ===============================================================

import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

# ---------- Optional deps ----------
def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False

HAS_LK   = _has("lightkurve")
HAS_AST  = _has("astropy")
HAS_MAST = _has("astroquery.mast")

if HAS_LK:
    import lightkurve as lk

if HAS_AST:
    from astropy.timeseries import BoxLeastSquares

if HAS_MAST:
    from astroquery.mast import Catalogs

warnings.filterwarnings("ignore")

# ---------- Paths (adjust if needed) ----------
BASE_DIR = Path(r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps")
DEFAULT_OUT = BASE_DIR / "Processed" / "LC_Model"

# ---------- XGB compatibility wrapper (for very old versions) ----------
class XGBBoosterWrapper:
    """Unifies predict/predict_proba when trained via xgboost.train()."""
    def __init__(self, booster, num_class: int):
        self.booster = booster
        self.num_class = num_class
    def _dmatrix(self, X):
        import xgboost as xgb
        return xgb.DMatrix(X)
    def predict(self, X):
        dm = self._dmatrix(X)
        if self.num_class <= 2:
            prob = self.booster.predict(dm)
            return (prob >= 0.5).astype(int)
        else:
            prob = self.booster.predict(dm)
            return np.argmax(prob, axis=1)
    def predict_proba(self, X):
        dm = self._dmatrix(X)
        prob = self.booster.predict(dm)
        if prob.ndim == 1:
            prob = np.vstack([1.0 - prob, prob]).T
        return prob

# ---------- Canonical features & labels ----------
CANON_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_impact", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg",
    "koi_srad", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
]

LABEL_CANON = {"CONFIRMED": 1, "CANDIDATE": 0, "FALSE POSITIVE": -1}
ALIASES = {
    "CONFIRMED": ["CONFIRMED", "CONFIRM", "CONFIRMEDED"],
    "CANDIDATE": ["CANDIDATE", "PC", "APC", "KP", "CP"],
    "FALSE POSITIVE": ["FALSE POSITIVE", "FP", "REFUTED"]
}

def normalize_label(name: str) -> str:
    n = str(name).upper().replace("_", " ").strip()
    for canon, al in ALIASES.items():
        for a in al:
            if a == n:
                return canon
    if "CONFIRM" in n:
        return "CONFIRMED"
    if "FALSE" in n or "FP" in n or "REFUTED" in n:
        return "FALSE POSITIVE"
    if "CANDIDATE" in n or n in {"PC","APC","KP","CP"}:
        return "CANDIDATE"
    return n

# ---------- Data access ----------
def fetch_tic_params(tic_id: str) -> Tuple[float, float, float]:
    """Return (teff, logg, radius) if available, else (0,0,0)."""
    if not HAS_MAST:
        return 0.0, 0.0, 0.0
    try:
        res = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
        if len(res) == 0:
            return 0.0, 0.0, 0.0
        row = res[0]
        teff = float(row["Teff"]) if row["Teff"] else 0.0
        logg = float(row["logg"]) if row["logg"] else 0.0
        rad  = float(row["rad"])  if row["rad"]  else 0.0
        return teff, logg, rad
    except Exception:
        return 0.0, 0.0, 0.0

def download_lightcurve_from_tic(tic_id: str, mission: str):
    if not HAS_LK:
        raise RuntimeError("lightkurve is required (pip install 'lightkurve[all]')")
    srch = lk.search_lightcurve(f"TIC {tic_id}", mission=mission.upper())
    if len(srch) == 0:
        raise FileNotFoundError(f"No light curve found for TIC {tic_id} ({mission})")
    lc = srch.download().remove_nans()
    try:
        lc = lc.flatten(window_length=301).normalize()
    except Exception:
        lc = lc.normalize()
    return lc.time.value, lc.flux.value

def read_lightcurve_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    tcol = next((c for c in ["time","Time","TIME","btjd","bjd","jd","t"] if c in df.columns), None)
    fcol = next((c for c in ["flux","Flux","FLUX","pdcsap_flux","sap_flux","f"] if c in df.columns), None)
    if not tcol or not fcol:
        raise ValueError("CSV must have 'time' and 'flux' (or btjd/pdcsap_flux).")
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    f = pd.to_numeric(df[fcol], errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    med = np.nanmedian(f)
    if not np.isfinite(med) or med == 0:
        raise ValueError("Flux median invalid or zero.")
    f = f / med
    return t, f

# ---------- FAST BLS (coarse→fine with binning & bounds) ----------
def fast_bls_features(time, flux,
                      min_period=None, max_period=None,
                      coarse_periods=1500, fine_periods=400,
                      oversample=3):
    """
    Return (period_days, duration_hours, depth_ppm, snr) using a 2-stage BLS.
    Includes a safety guard to ensure min_period > max(duration_grid).
    Robust SNR calculation for Astropy versions without res.snr.
    """
    if not HAS_AST:
        raise RuntimeError("astropy is required for BLS (pip install astropy)")

    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < 50:
        raise ValueError("Not enough points for BLS (need ≥ ~50).")

    # Normalize & clip outliers
    f_med = np.nanmedian(f)
    if not np.isfinite(f_med) or f_med == 0:
        raise ValueError("Flux median invalid or zero after filtering.")
    f = f / f_med
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med)) + 1e-12
    f = np.clip(f, med - 8*mad, med + 8*mad)

    # 10-min binning for speed if very dense
    bin_dt = 10.0 / (24.0 * 60.0)
    if len(t) > 5000:
        t0 = t.min()
        bins = np.floor((t - t0) / bin_dt).astype(int)
        df = pd.DataFrame({"b": bins, "t": t, "f": f})
        g = df.groupby("b", as_index=False).agg({"t": "mean", "f": "median"})
        t, f = g["t"].to_numpy(), g["f"].to_numpy()

    baseline = t.max() - t.min()
    if baseline <= 0:
        raise ValueError("Time span is zero.")

    # Duration grid: 0.5h..6h (tighter to avoid clashes and speed up)
    dur_grid = np.linspace(0.5/24, 6/24, 20)
    longest = float(dur_grid.max())

    # Default period bounds with safety guard
    if min_period is None:
        min_period = max(0.25, 2 * (np.median(np.diff(np.sort(t))) + 1e-6), longest * 1.2)
    else:
        min_period = max(min_period, longest * 1.2)

    if max_period is None:
        max_period = max(1.0, min(0.8 * baseline, 50.0))

    if min_period >= max_period:
        # fallback small window consistent with guard
        min_period, max_period = max(longest * 1.2, 0.7), max(longest * 1.2 + 0.7, 2.5)

    bls = BoxLeastSquares(t, f)

    # Stage 1: coarse scan
    periods_coarse = np.linspace(min_period, max_period, int(coarse_periods))
    res1 = bls.power(periods_coarse, dur_grid, oversample=oversample)
    i1 = int(np.nanargmax(res1.power))
    p1 = float(res1.period[i1])

    # Stage 2: fine around p1
    half = max(0.03 * p1, 0.05)
    pmin2 = max(min_period, p1 - half)
    pmax2 = min(max_period, p1 + half)
    if pmax2 <= pmin2:
        pmin2, pmax2 = max(min_period, p1 * 0.98), min(max_period, p1 * 1.02)
    periods_fine = np.linspace(pmin2, pmax2, int(fine_periods))
    res2 = bls.power(periods_fine, dur_grid, oversample=oversample)
    i2 = int(np.nanargmax(res2.power))

    period = float(res2.period[i2])
    duration_days = float(res2.duration[i2])
    depth = float(res2.depth[i2])  # fractional drop (e.g., 0.001 = 0.1%)
    # ---- Robust SNR (handle Astropy versions without res.snr) ----
    snr = np.nan
    try:
        # Newer Astropy may have .snr
        snr = float(res2.snr[i2])  # type: ignore[attr-defined]
    except Exception:
        # Fallback 1: depth / depth_err
        try:
            depth_err = float(res2.depth_err[i2])  # type: ignore[attr-defined]
            if np.isfinite(depth_err) and depth_err > 0:
                snr = depth / depth_err
        except Exception:
            pass
        # Fallback 2: use normalized BLS power as proxy
        if not np.isfinite(snr):
            pwr = res2.power
            denom = np.nanstd(pwr) + 1e-12
            snr = float((pwr[i2] - np.nanmean(pwr)) / denom)

    # Final guards
    if not np.isfinite(snr):
        snr = 0.0
    if duration_days <= 0:
        duration_days = 0.5 / 24.0  # 0.5h fallback

    duration_hours = duration_days * 24.0
    depth_ppm = max(0.0, depth) * 1e6
    return period, duration_hours, depth_ppm, snr

def build_feature_row(period_days, duration_hours, depth_ppm, snr,
                      teff=0.0, logg=0.0, srad=0.0):
    return {
        "koi_period": period_days,
        "koi_duration": duration_hours,
        "koi_depth": depth_ppm,
        "koi_impact": 0.0,
        "koi_prad": 0.0,
        "koi_teq": 0.0,
        "koi_insol": 0.0,
        "koi_model_snr": snr,
        "koi_steff": teff,
        "koi_slogg": logg,
        "koi_srad": srad,
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 0,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0
    }

# ---------- Model builders ----------
def build_model(name: str, seed: int, class_weight=None):
    name = name.lower()
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=400, random_state=seed, n_jobs=-1, class_weight=class_weight
        )
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=4000, learning_rate=0.03, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, n_jobs=-1
        )
    if name == "xgb":
        return "XGB_PLACEHOLDER"
    raise ValueError("Unknown model: use xgb | lgbm | rf")

def train_xgb(X_train, y_train, X_val=None, y_val=None, seed=42, cw=None):
    import xgboost as xgb
    # Try sklearn API first
    try:
        sample_weight = None
        if cw is not None:
            sw_map = {int(k): float(v) for k, v in cw.items()}
            sample_weight = np.vectorize(lambda c: sw_map[int(c)])(y_train)

        model = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=seed, n_jobs=-1, eval_metric="mlogloss"
        )
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=100 if eval_set else None
        )
        return model
    except TypeError:
        # Fallback: native Booster path (works on very old versions)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val) if X_val is not None else None
        params = {
            "seed": seed, "eta": 0.03, "max_depth": 6,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "tree_method": "hist", "eval_metric": "mlogloss",
            "objective": "multi:softprob" if len(np.unique(y_train)) > 2 else "binary:logistic",
        }
        if len(np.unique(y_train)) > 2:
            params["num_class"] = len(np.unique(y_train))
        evals = [(dtrain, "train")]
        if dvalid is not None:
            evals.append((dvalid, "valid"))
        booster = xgb.train(
            params=params, dtrain=dtrain, evals=evals,
            num_boost_round=2000, early_stopping_rounds=100 if dvalid is not None else 0,
            verbose_eval=False
        )
        return XGBBoosterWrapper(booster, num_class=params.get("num_class", 2))

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Train a model from light curves (download or CSV).")
    # Inputs (choose one or both)
    ap.add_argument("--tic_ids", nargs="*", default=None, help="List of TIC IDs (TESS/Kepler) to download")
    ap.add_argument("--csv_glob", default=None, help=r'Glob of local CSVs with time/flux, e.g. "Data\LightCurves\*.csv"')

    # Labels (optional, supervised training)
    ap.add_argument("--labels_csv", default=None,
                    help='CSV with columns: source,label (source is TIC<ID> or filename stem)')
    ap.add_argument("--label_map_json", default=None,
                    help='Optional JSON mapping label string -> int (default uses CONFIRMED=1, CANDIDATE=0, FALSE POSITIVE=-1)')

    # BLS controls
    ap.add_argument("--min_period", type=float, default=None, help="Min period (days)")
    ap.add_argument("--max_period", type=float, default=None, help="Max period (days)")
    ap.add_argument("--coarse", type=int, default=1200, help="Coarse period grid size")
    ap.add_argument("--fine", type=int, default=350, help="Fine period grid size")
    ap.add_argument("--mission", default="tess", help="tess or kepler for TIC download")

    # Training controls
    ap.add_argument("--model", default="xgb", choices=["xgb","rf","lgbm"], help="Classifier to train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(DEFAULT_OUT), help="Output directory")
    ap.add_argument("--features_only", action="store_true", help="Only build features (no training)")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load labels if provided
    labels_df = None
    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        if not {"source","label"}.issubset(labels_df.columns):
            raise ValueError("labels_csv must have columns: source,label")
        labels_df["label"] = labels_df["label"].apply(normalize_label)

    # Make sources list
    sources = []  # list of tuples: (source_name, get_time_flux_callable, stellar_params)
    # TIC sources
    if args.tic_ids:
        for tid in args.tic_ids:
            sname = f"TIC{str(tid).strip()}"
            def make_loader(tic_id):
                return lambda: download_lightcurve_from_tic(str(tic_id), args.mission)
            teff, logg, srad = fetch_tic_params(str(tid))
            sources.append((sname, make_loader(tid), (teff, logg, srad)))
    # CSV glob
    if args.csv_glob:
        from glob import glob
        for path in glob(args.csv_glob):
            p = Path(path)
            sname = p.stem
            sources.append((sname, lambda p=p: read_lightcurve_csv(p), (0.0, 0.0, 0.0)))

    if not sources:
        raise ValueError("No sources provided. Use --tic_ids and/or --csv_glob.")

    rows = []
    for sname, loader, (teff, logg, srad) in sources:
        try:
            time, flux = loader()
            p, d_h, depth_ppm, snr = fast_bls_features(
                time, flux,
                min_period=args.min_period,
                max_period=args.max_period,
                coarse_periods=args.coarse,
                fine_periods=args.fine,
                oversample=3,
            )
            feat = build_feature_row(p, d_h, depth_ppm, snr, teff, logg, srad)
            row = {"source": sname, **feat}
            rows.append(row)
            print(f"✅ {sname}: P={p:.4f} d, dur={d_h:.2f} h, depth={depth_ppm:.0f} ppm, SNR={snr:.2f}")
        except Exception as e:
            print(f"⚠️ Skipping {sname}: {e}")

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        raise RuntimeError("No features extracted. Check inputs or BLS bounds.")

    # Save raw features
    feat_path = outdir / "lc_features.csv"
    features_df.to_csv(feat_path, index=False)
    print(f"\n📝 Saved features -> {feat_path}")

    # Attach labels if available
    y = None
    if labels_df is not None:
        merged = features_df.merge(labels_df, how="left", left_on="source", right_on="source")
        if merged["label"].isna().any():
            missing = merged[merged["label"].isna()]["source"].tolist()
            print(f"⚠️ Missing labels for sources (will drop for supervised training): {missing}")
        train_df = merged.dropna(subset=["label"]).copy()
        if train_df.empty:
            print("⚠️ No labeled rows found. Will skip training.")
        else:
            features_df = train_df
            y = features_df["label"].apply(normalize_label).tolist()

    # Build X matrix in correct order
    for col in CANON_FEATURES:
        if col not in features_df.columns:
            features_df[col] = 0.0
    X = features_df[CANON_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # feature metadata
    meta = {
        "feature_cols": CANON_FEATURES,
        "label_map": LABEL_CANON  # canonical suggestion for new models
    }
    (outdir / "feature_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # If only features requested, stop here
    if args.features_only or y is None:
        print("\nℹ️ features_only=True or no labels provided — skipping training.")
        return

    # Encode labels to ints using canonical mapping (or fallback)
    y_arr = []
    for lbl in y:
        canon = normalize_label(lbl)
        y_arr.append(LABEL_CANON.get(canon, 0))
    y = np.array(y_arr, dtype=int)

    # Simple split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    # Train model
    model_name = args.model.lower()
    if model_name == "xgb":
        model = train_xgb(X_train, y_train, X_test, y_test, seed=args.seed, cw=class_weight)
    else:
        model = build_model(model_name, seed=args.seed, class_weight=class_weight if model_name=="rf" else None)
        if model_name == "lgbm":
            import lightgbm as lgb
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric="multi_logloss",
                      callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        else:
            model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "report": classification_report(y_test, y_pred)
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n📊 Metrics:")
    print(json.dumps(metrics, indent=2))

    # Save model
    joblib.dump(model, outdir / "model.pkl")
    print(f"💾 Model saved -> {outdir / 'model.pkl'}")

    # Quick scored preview on full feature table
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        features_df["Pred_ClassID"] = np.argmax(proba, axis=1)
        features_df["Pred_ProbMax"] = proba.max(axis=1)
    else:
        pred = model.predict(X)
        features_df["Pred_ClassID"] = pred
        features_df["Pred_ProbMax"] = np.nan

    preview_path = outdir / "scored_preview.csv"
    features_df[["source"] + CANON_FEATURES + ["Pred_ClassID","Pred_ProbMax"]].to_csv(preview_path, index=False)
    print(f"📝 Scored preview -> {preview_path}")

if __name__ == "__main__":
    main()
