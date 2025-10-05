# -*- coding: utf-8 -*-
"""
Space Invaders — SH -> FITS -> Features -> Predictions (No-CLI, Click-to-Run)

Save as:
    Code/run_sh_to_predictions.py

What it does:
- Lets you choose a TESS *.sh* bulk-download script (e.g., tesscurl_sector_96_lc.sh)
- Parses URLs, downloads up to MAX_FILES FITS light curves
- Extracts features (period, duration, depth, snr) via BLS
- Adds optional stellar context if present (Teff, logg, R*, M*)
- Auto-aligns feature names to your trained model's expected columns
- Outputs CSV + JSON with predictions for your React app

Outputs (next to the *.sh*):
- <sh_name>_features.csv
- <sh_name>_predictions.csv
- <sh_name>_predictions.json
"""

import os
import re
import sys
import json
import math
import time
import joblib
import shutil
import signal
import zipfile
import requests
import warnings
import tempfile
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Astronomy libs
import lightkurve as lk
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------
# USER TUNABLES
# ------------------------------
MAX_FILES = 300               # how many FITS to process from the .sh
TIMEOUT   = 30                # seconds per HTTP download
BLS_MIN_P = 0.5               # days
BLS_MAX_P = 30.0              # days
BLS_COARSE = 1000             # coarse grid steps
BLS_FINE   = 300              # fine grid steps around best peak
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Processed", "LC_Model_Pro", "lightcurve_model.pkl"
)

# ------------------------------
# Helpers
# ------------------------------

def pick_file(title="Choose a file", filetypes=(("All files", "*.*"),)):
    """Simple GUI file picker (no CLI)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path
    except Exception:
        # Fallback to input if Tk is not available
        return input(f"{title}: ").strip()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_sh_to_urls(sh_path: str) -> List[str]:
    """Extract HTTPS URLs from a tesscurl_*.sh list."""
    urls = []
    with open(sh_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "https://" in line:
                # tolerate quotes & spaces
                found = re.findall(r"https://\S+", line)
                for u in found:
                    # trim trailing punctuation
                    u = u.strip('\'" )(').rstrip("\\")
                    urls.append(u)
    urls = [u for u in urls if u.lower().endswith((".fits", ".fits.gz"))]
    return urls

def download_file(url: str, out_dir: str, timeout: int = TIMEOUT) -> Optional[str]:
    """Download URL to out_dir; return local path or None on failure."""
    ensure_dir(out_dir)
    local = os.path.join(out_dir, os.path.basename(url))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return local
    except Exception:
        if os.path.exists(local):
            try: os.remove(local)
            except Exception: pass
        return None

def load_time_flux_from_fits(path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Open a TESS/Kepler light curve FITS and return (time, flux, meta).
    Prefer Lightkurve's LightCurve + normalized PDCSAP if available.
    """
    try:
        lkobj = lk.open(path)  # LightCurve or LightCurveFile
        # Get a LightCurve
        if hasattr(lkobj, "to_lightcurve"):
            lc = lkobj.to_lightcurve()
        else:
            lc = lkobj  # already a LightCurve

        # Prefer PDCSAP if present; fallback to whatever the LC has
        cols = [c.lower() for c in (lc.columns() if hasattr(lc, "columns") else [])]
        if "pdcsap_flux" in cols:
            pass
        elif hasattr(lkobj, "PDCSAP_FLUX") and lkobj.PDCSAP_FLUX is not None:
            # older lightkurve
            lc = lkobj.PDCSAP_FLUX

        # Clean & normalize
        lc = lc.remove_nans().remove_outliers(sigma=6.0).normalize(unit="relative")
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)
        meta = dict(getattr(lc, "meta", {}))
        return time, flux, meta

    except Exception as e:
        # Fallback to manual FITS read
        try:
            with fits.open(path, mode='readonly') as hdul:
                for hdu in hdul:
                    if isinstance(hdu, fits.BinTableHDU):
                        cols = [c.lower() for c in hdu.columns.names]
                        if "time" in cols and ("pdcsap_flux" in cols or "sap_flux" in cols or "flux" in cols):
                            data = hdu.data
                            time = np.array(data["TIME"], dtype=float)
                            if "pdcsap_flux" in cols:
                                flux_col = "PDCSAP_FLUX"
                            elif "sap_flux" in cols:
                                flux_col = "SAP_FLUX"
                            else:
                                flux_col = "FLUX"
                            flux = np.array(data[flux_col], dtype=float)
                            mask = np.isfinite(time) & np.isfinite(flux)
                            time, flux = time[mask], flux[mask]
                            flux = flux / np.nanmedian(flux)
                            meta = dict(hdu.header)
                            return time, flux, meta
        except Exception:
            pass
        raise RuntimeError(f"Could not read light curve from {os.path.basename(path)}; {e}")

def bls_features(time: np.ndarray, flux: np.ndarray,
                 pmin=BLS_MIN_P, pmax=BLS_MAX_P,
                 coarse=BLS_COARSE, fine=BLS_FINE) -> Tuple[float, float, float, float]:
    """
    Compute BLS and return (period_days, duration_hours, depth_ppm, snr).
    """
    mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]
    y = flux[mask]
    if len(t) < 50:
        raise ValueError("Too few points for BLS.")

    bls = BoxLeastSquares(t, 1.0 - y)  # invert so dips become "peaks"
    periods = np.linspace(pmin, pmax, int(coarse))
    durations = np.linspace(0.5/24, 8.0/24, 30)  # 0.5h..8h

    res = bls.power(periods, durations)
    k = np.argmax(res.power)
    p0, d0 = res.period[k], res.duration[k]

    # refine near best
    p_grid = np.linspace(max(p0*0.9, pmin), min(p0*1.1, pmax), int(fine))
    d_grid = np.linspace(max(d0*0.5, 0.25/24), min(d0*1.5, 10.0/24), 25)
    res2 = bls.power(p_grid, d_grid)
    k2 = np.argmax(res2.power)

    period = float(res2.period[k2])
    duration = float(res2.duration[k2])
    depth = float(res2.depth[k2])      # fractional
    snr = float(res2.power[k2])

    duration_h = duration * 24.0
    depth_ppm = depth * 1e6
    return period, duration_h, depth_ppm, snr

def stellar_meta_from_header(meta: Dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Try to extract basic stellar params if available.
    Returns (Teff[K], logg[cgs], R*[Rsun], M*[Msun]) or (None,...)
    """
    if meta is None:
        return None, None, None, None
    # Common header keys in MAST products
    teff = meta.get("TEFF", meta.get("TEFFVAL", None))
    logg = meta.get("LOGG", meta.get("LOGGVAL", None))
    rstar = meta.get("RADIUS", meta.get("RSTAR", None))
    mstar = meta.get("MASS", meta.get("MSTAR", None))

    def asfloat(x):
        try: return float(x)
        except Exception: return None

    return asfloat(teff), asfloat(logg), asfloat(rstar), asfloat(mstar)

def identify_target_from_path(path: str) -> str:
    """Heuristic label: TIC######## or KIC#### from filename."""
    fname = os.path.basename(path)
    m = re.search(r"(TIC[_ ]?\d+)", fname, re.IGNORECASE)
    if m: return "TIC " + re.sub(r"\D", "", m.group(0))
    m = re.search(r"(KIC[_ ]?\d+)", fname, re.IGNORECASE)
    if m: return "KIC " + re.sub(r"\D", "", m.group(0))
    return os.path.splitext(fname)[0]

def estimate_radius_re(depth_ppm: float, star_rad_rsun: Optional[float]) -> Optional[float]:
    """
    Approximate planet radius (Earth radii) from transit depth (ppm) and stellar radius (Rsun).
    Rp/Rs ~ sqrt(depth); Rp(Rearth) ~ sqrt(depth) * Rsun/R_earth
    1 Rsun ≈ 109.1 R_earth
    """
    if depth_ppm is None or not np.isfinite(depth_ppm):
        return None
    if star_rad_rsun is None or not np.isfinite(star_rad_rsun):
        return None
    depth_frac = depth_ppm / 1e6
    if depth_frac <= 0:
        return None
    return float(np.sqrt(depth_frac) * star_rad_rsun * 109.1)

def estimate_semimajor_axis_au(period_days: float, star_mass_msun: Optional[float]) -> Optional[float]:
    """
    Kepler's third law (approx):
    a(AU) ~ ( Mstar * (P_year)^2 )^(1/3), with P_year = P_days / 365.25
    """
    if period_days is None or not np.isfinite(period_days):
        return None
    if star_mass_msun is None or not np.isfinite(star_mass_msun):
        star_mass_msun = 1.0
    P_year = period_days / 365.25
    return float((star_mass_msun * (P_year**2))**(1/3))

def build_feature_row(ident: str, meta: Dict,
                      p: float, dur_h: float, depth_ppm: float, snr: float) -> Dict:
    teff, logg, rstar, mstar = stellar_meta_from_header(meta)
    row = {
        "source": ident,
        "origin": "download",
        "identifier": ident,

        # long names
        "period_days": p,
        "duration_hours": dur_h,
        "depth_ppm": depth_ppm,
        "snr": snr,
        "teff_k": teff if teff is not None else np.nan,
        "logg_cgs": logg if logg is not None else np.nan,
        "star_rad_rsun": rstar if rstar is not None else np.nan,
        "star_mass_msun": mstar if mstar is not None else np.nan,

        # short names that many of your models expect
        "period": p,
        "duration": dur_h,
        "depth": depth_ppm,
    }

    # Friendly estimates
    row["Est_Radius_Re"] = estimate_radius_re(depth_ppm, rstar)
    row["Est_SemiMajorAxis_AU"] = estimate_semimajor_axis_au(p, mstar)
    row["Est_a_over_Rstar"] = (row["Est_SemiMajorAxis_AU"] * 215.032) / rstar if (row["Est_SemiMajorAxis_AU"] and rstar and rstar > 0) else np.nan
    row["Period_vs_Earth"] = p / 365.25 if p else np.nan
    return row

def predict_df(model, df_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Align features to model expectations and predict.
    Works whether model expects ['period','duration','depth','snr'] (short)
    or long names ['period_days','duration_hours','depth_ppm','snr',...].
    """
    df = df_feats.copy()

    # Ensure short aliases exist
    if "period" not in df.columns and "period_days" in df.columns:
        df["period"] = df["period_days"]
    if "duration" not in df.columns and "duration_hours" in df.columns:
        df["duration"] = df["duration_hours"]
    if "depth" not in df.columns and "depth_ppm" in df.columns:
        # If your model was trained on fractional depth use: df["depth"] = df["depth_ppm"]/1e6
        df["depth"] = df["depth_ppm"]

    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        # Fall back to short set used in earlier LC models
        expected = np.array(["period", "duration", "depth", "snr"])

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Model expects {list(expected)}; missing {missing}")

    X = df[list(expected)].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1, 2]))
    idx_to_label = {}
    if set(classes) == {-1, 0, 1}:
        for i, c in enumerate(classes):
            idx_to_label[i] = {-1: "FALSE_POSITIVE", 0: "CANDIDATE", 1: "CONFIRMED"}[c]
    else:
        default_map = {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE_POSITIVE"}
        for i, c in enumerate(classes):
            idx_to_label[i] = default_map.get(c, f"class_{c}")

    pred_idx = np.argmax(proba, axis=1)
    pred_lab = [idx_to_label[i] for i in pred_idx]

    out = df_feats.copy()
    out["Pred_Class_Index"] = pred_idx
    out["Pred_Label"] = pred_lab
    for i in range(proba.shape[1]):
        out[f"Prob_class_{i}"] = proba[:, i]
    return out

def process_sh_to_predictions(sh_path: str,
                              model_path: Optional[str] = None,
                              limit: int = MAX_FILES) -> Tuple[str, str, str]:
    """
    Main pipeline:
    - parse .sh, download subset of FITS
    - extract features
    - load model, score, save CSV/JSON
    Returns (features_csv, predictions_csv, predictions_json)
    """
    base_dir = os.path.dirname(sh_path)
    base_name = os.path.splitext(os.path.basename(sh_path))[0]

    print(f"🔎 Reading list: {os.path.basename(sh_path)}")
    urls = parse_sh_to_urls(sh_path)
    if not urls:
        raise RuntimeError("No FITS URLs found in the .sh file.")
    urls = urls[:limit]
    print(f"🌐 {len(urls)} URLs to fetch/process")

    dl_dir = os.path.join(base_dir, f"{base_name}_fits")
    ensure_dir(dl_dir)

    features = []
    for url in tqdm(urls, desc="⬇️  Download + Extract", unit="file"):
        fp = download_file(url, dl_dir, timeout=TIMEOUT)
        if fp is None:
            continue
        try:
            t, f, meta = load_time_flux_from_fits(fp)
            period, dur_h, depth_ppm, snr = bls_features(t, f)
            ident = identify_target_from_path(fp)
            row = build_feature_row(ident, meta, period, dur_h, depth_ppm, snr)
            features.append(row)
        except Exception:
            # skip bad files
            continue

    if not features:
        raise RuntimeError("No usable light curves extracted; check the list or bounds.")

    df = pd.DataFrame(features)
    feats_csv = os.path.join(base_dir, f"{base_name}_features.csv")
    df.to_csv(feats_csv, index=False)
    print(f"📝 Saved features -> {feats_csv}")

    # --------------- Prediction ---------------
    if model_path is None or not os.path.exists(model_path):
        print("📦 Choose your trained light-curve model (.pkl)")
        model_path = pick_file("Pick model (.pkl)", (("Pickle model", "*.pkl"), ("All files", "*.*")))
        if not model_path:
            raise RuntimeError("No model selected.")

    print("🤖 Loading model and predicting …")
    model = joblib.load(model_path)
    pred_df = predict_df(model, df)

    # Save outputs
    pred_csv = os.path.join(base_dir, f"{base_name}_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"✅ Saved predictions CSV -> {pred_csv}")

    # React-friendly JSON (flat list of dicts)
    json_path = os.path.join(base_dir, f"{base_name}_predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pred_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"✅ Saved predictions JSON -> {json_path}")

    return feats_csv, pred_csv, json_path

# ------------------------------
# Entry point (no arguments required)
# ------------------------------

def gui_main():
    print("🚀 Space Invaders — SH→Predictions")
    print("Pick a TESS *.sh* file (e.g., tesscurl_sector_96_lc.sh)")
    sh_path = pick_file("Choose tesscurl_*.sh", (("Shell list", "*.sh"), ("All files", "*.*")))
    if not sh_path or not os.path.exists(sh_path):
        print("No file chosen. Exiting.")
        return

    # Default model is your LC pro model; if missing, we'll ask.
    model_path = DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else None

    try:
        feats_csv, pred_csv, json_path = process_sh_to_predictions(
            sh_path=sh_path,
            model_path=model_path,
            limit=MAX_FILES
        )
        print("\n🎉 Done.")
        print(f"Features CSV: {feats_csv}")
        print(f"Predictions CSV: {pred_csv}")
        print(f"Predictions JSON: {json_path}")
    except Exception as e:
        print(f"\n💥 Error: {e}")

if __name__ == "__main__":
    gui_main()
