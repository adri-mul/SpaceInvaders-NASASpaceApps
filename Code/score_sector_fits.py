# score_sector_fits.py
# One-pass: read TESS LC FITS -> extract BLS features -> predict with your lightcurve model -> CSV

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd

from astropy.timeseries import BoxLeastSquares
from astropy import units as u
import lightkurve as lk
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Processed\LC_Model_Pro\lightcurve_model.pkl"
DEFAULT_INDIR = r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Data\TESS\S96_lc"
DEFAULT_OUTCSV = r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Predictions\sector96_scored.csv"

# ---------- feature extraction via BLS ----------
def clean_lightcurve(lc: lk.LightCurve, min_points=300):
    lc = lc.remove_nans().remove_outliers(sigma=5)
    if len(lc.time) < min_points:
        return None
    # Normalize & flatten to remove long-term trends
    try:
        lc = lc.normalize().flatten(window_length=401)
    except Exception:
        lc = lc.normalize()
    return lc

def bls_features(time_bkjd, flux, min_period=0.5, max_period=20.0):
    time = np.array(time_bkjd, dtype=float)
    flux = np.array(flux, dtype=float)

    # Convert to days and unitless
    t = time
    y = flux / np.nanmedian(flux)
    y = y - np.nanmedian(y)
    w = np.isfinite(t) & np.isfinite(y)
    t, y = t[w], y[w]
    if len(t) < 200:
        raise ValueError("too few points after cleaning")

    bls = BoxLeastSquares(t, y)
    # Coarse-to-fine grid for speed+robustness
    periods = np.linspace(min_period, max_period, 1000)
    durations = np.linspace(0.5/24, 8/24, 20)  # 0.5h to 8h

    res = bls.power(periods, durations)
    i = np.argmax(res.power)
    best_period = res.period[i]
    best_depth = res.depth[i]
    best_duration = res.duration[i]
    snr = (res.depth_snr[i] if hasattr(res, "depth_snr") else res.power[i])

    # sanity to keep things positive
    best_depth_ppm = max(0.0, float(best_depth*1e6))
    duration_hours = max(0.0, float(best_duration*24.0))
    period_days = max(0.0, float(best_period))
    snr = float(max(0.0, snr))

    return {
        "period_days": period_days,
        "duration_hours": duration_hours,
        "depth_ppm": best_depth_ppm,
        "snr": snr,
    }

# ---------- model wrapper ----------
def load_model(path):
    model = joblib.load(path)
    # Must expose predict_proba; if model lacks it, make a uniform fallback
    if not hasattr(model, "predict_proba"):
        def _uniform(X):
            n = len(X)
            return np.ones((n, 3)) / 3.0
        model.predict_proba = _uniform  # type: ignore
    return model

def score_features(df_feat, model):
    needed = ["period_days", "duration_hours", "depth_ppm", "snr"]
    for c in needed:
        if c not in df_feat.columns:
            df_feat[c] = np.nan
    X = df_feat[needed].to_numpy(dtype=float)
    proba = model.predict_proba(X)
    if proba.shape[1] == 3:
        # class order assumed [CANDIDATE, CONFIRMED, FALSE_POSITIVE]
        pred_idx = np.argmax(proba, axis=1)
        map_idx_to_label = {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE_POSITIVE"}
        df_feat["Pred_Class_Index"] = pred_idx
        df_feat["Pred_Label"] = [map_idx_to_label.get(i, "UNKNOWN") for i in pred_idx]
        df_feat["Prob_CANDIDATE"] = proba[:, 0]
        df_feat["Prob_CONFIRMED"] = proba[:, 1]
        df_feat["Prob_FALSE_POSITIVE"] = proba[:, 2]
    else:
        # unknown shape -> uniform
        df_feat["Pred_Class_Index"] = 0
        df_feat["Pred_Label"] = "CANDIDATE"
        df_feat["Prob_CANDIDATE"] = 1/3
        df_feat["Prob_CONFIRMED"] = 1/3
        df_feat["Prob_FALSE_POSITIVE"] = 1/3
    return df_feat

# ---------- IO ----------
def find_fits(indir):
    fits = []
    for root, _, files in os.walk(indir):
        for f in files:
            if f.lower().endswith(".fits") and "lc" in f.lower():
                fits.append(os.path.join(root, f))
    return sorted(fits)

def parse_identifiers_from_filename(path):
    # Example filename contains TIC-xxxxx or similar; do best-effort
    name = os.path.basename(path)
    tic = None
    m = re.search(r"TIC[_-]?\s*(\d+)", name, flags=re.IGNORECASE)
    if m: tic = m.group(1)
    return tic

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Score TESS sector LC FITS with your lightcurve model.")
    ap.add_argument("--indir", default=DEFAULT_INDIR, help="Folder containing downloaded LC FITS (from tesscurl script).")
    ap.add_argument("--model", default=MODEL_PATH, help="Path to your lightcurve model .pkl")
    ap.add_argument("--out", default=DEFAULT_OUTCSV, help="Output CSV path")
    ap.add_argument("--min_period", type=float, default=0.5, help="BLS min period [days]")
    ap.add_argument("--max_period", type=float, default=20.0, help="BLS max period [days]")
    ap.add_argument("--limit", type=int, default=300, help="Max number of FITS to process (safety)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_model(args.model)
    files = find_fits(args.indir)
    if len(files) == 0:
        print(f"❌ No FITS found in: {args.indir}")
        return
    if args.limit and len(files) > args.limit:
        files = files[:args.limit]
        print(f"ℹ️ Limiting to first {args.limit} files.")

    rows = []
    processed = 0
    for i, fp in enumerate(files, 1):
        try:
            lcf = lk.read(fp)  # LightCurveFile
            # Prefer PDCSAP if available
            if hasattr(lcf, "PDCSAP_FLUX") and lcf.PDCSAP_FLUX is not None:
                lc = lcf.PDCSAP_FLUX
            elif hasattr(lcf, "SAP_FLUX") and lcf.SAP_FLUX is not None:
                lc = lcf.SAP_FLUX
            else:
                # fallback: try to convert generically
                lc = lcf.to_lightcurve()
            lc = clean_lightcurve(lc)
            if lc is None:
                continue

            feats = bls_features(
                lc.time.value if hasattr(lc.time, "value") else lc.time, 
                lc.flux.value if hasattr(lc.flux, "value") else lc.flux,
                min_period=args.min_period,
                max_period=args.max_period,
            )
            feats["source_file"] = fp
            feats["tic_id"] = parse_identifiers_from_filename(fp) or ""
            rows.append(feats)
            processed += 1
            if processed % 25 == 0:
                print(f"…processed {processed} light curves")
        except Exception as e:
            # Skip problematic targets; sector sets can include odd files
            # print(f"Skip {os.path.basename(fp)}: {e}")
            continue

    if len(rows) == 0:
        print("❌ No features extracted. Try widening period range or different files.")
        return

    df = pd.DataFrame(rows)
    df_scored = score_features(df, model)
    # Optional derived columns for the demo:
    # crude planet radius estimate from depth (Rp/R*) ≈ sqrt(depth), if R* unknown assume 1 Rsun:
    df_scored["Est_Radius_Re"] = np.sqrt(np.clip(df_scored["depth_ppm"] / 1e6, 0, None)) * 109.1  # Earth radii if R*~Rsun
    df_scored["Period_vs_Earth"] = df_scored["period_days"] / 365.25

    # Order columns nicely
    cols = [
        "tic_id", "source_file",
        "period_days", "duration_hours", "depth_ppm", "snr",
        "Pred_Label", "Prob_CANDIDATE", "Prob_CONFIRMED", "Prob_FALSE_POSITIVE",
        "Est_Radius_Re", "Period_vs_Earth"
    ]
    for c in cols:
        if c not in df_scored.columns:
            df_scored[c] = np.nan
    df_scored = df_scored[cols].sort_values("Prob_CONFIRMED", ascending=False)

    df_scored.to_csv(args.out, index=False)
    print(f"✅ Saved predictions → {args.out}")
    print(df_scored.head(10).to_string(index=False))

if __name__ == "__main__":
    main()