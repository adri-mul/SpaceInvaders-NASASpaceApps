# ============================================================
# Space Invaders — Analyze a Light-Curve CSV and Classify
# Usage (Option 1: interactive):
#   python Code/analyze_lightcurve_csv.py
#   (paste your CSV path when prompted)
#
# Usage (Option 2: CLI):
#   python Code/analyze_lightcurve_csv.py --csv "path\to\lightcurve.csv" \
#       --model "C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Processed\LC_Model_Pro\lightcurve_model.pkl"
#
# Output:
#   <input_dir>\<input_basename>_prediction.csv
# ============================================================

import os
import argparse
import warnings
import numpy as np
import pandas as pd

# Astropy BLS for transit search
from astropy.timeseries import BoxLeastSquares
from astropy.stats import sigma_clip

# For model loading
import joblib

# Optional but useful
try:
    from scipy.signal import savgol_filter
    HAVE_SAVGOL = True
except Exception:
    HAVE_SAVGOL = False


# ----------------------- Helpers -----------------------

POSSIBLE_TIME = [
    "time", "bjd", "BJD", "bjdt", "jd", "JD", "t", "BTJD", "btjd", "TIME"
]
POSSIBLE_FLUX = [
    "flux", "FLUX", "sap_flux", "PDCSAP_FLUX", "pdcsap_flux", "SAP_FLUX",
    "flux_norm", "f", "relative_flux"
]
# Optional star meta columns if present in the CSV
STAR_META = {
    "teff_k": ["teff", "teff_k", "stellar_teff_k", "T_eff", "st_teff"],
    "logg":   ["logg", "logg_cgs", "st_logg"],
    "rstar":  ["rstar", "r_star", "star_radius_rsun", "R_star", "st_rad", "rstar_rsun"],
    "mstar":  ["mstar", "m_star", "star_mass_msun", "st_mass", "mstar_msun"]
}

def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        # case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def extract_star_params(df: pd.DataFrame):
    # Try to read scalar star params (single values) if present in the CSV
    out = {"teff_k": None, "logg": None, "rstar": None, "mstar": None}
    for key, names in STAR_META.items():
        col = find_column(df, names)
        if col is not None:
            # If it's a column constant or a single-row file, pick median
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = pd.to_numeric(df[col], errors="coerce")
            v = np.nanmedian(val.values) if len(val) else np.nan
            out[key] = float(v) if np.isfinite(v) else None
    return out

def preprocess_flux(time, flux):
    # 1) remove NaNs
    mask = np.isfinite(time) & np.isfinite(flux)
    t = np.array(time[mask], dtype=float)
    f = np.array(flux[mask], dtype=float)

    # 2) sigma-clip
    f = sigma_clip(f, sigma=5, maxiters=2, masked=False)

    # 3) normalize to median
    med = np.nanmedian(f)
    if not np.isfinite(med) or med == 0:
        med = 1.0
    f = f / med

    # 4) simple detrend (optional)
    if HAVE_SAVGOL and len(f) >= 51:
        try:
            # window length must be odd and < len
            wl = min(len(f) // 5 * 2 + 1, 301)
            wl = max(51, wl)
            if wl >= len(f):  # fallback
                wl = len(f) - 1 if (len(f) - 1) % 2 == 1 else len(f) - 2
            trend = savgol_filter(f, window_length=max(5, wl), polyorder=2, mode="interp")
            f = f / np.where(trend == 0, 1.0, trend)
        except Exception:
            pass

    return t, f

def bls_features(time, flux, min_period=0.3, max_period=30.0, n_coarse=1000, n_fine=300):
    """Return period [d], duration [h], depth [ppm], snr."""
    # frequency grid
    duration_grid = np.linspace(0.5/24.0, 8.0/24.0, 25)  # 0.5h to 8h
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(min_period, max_period, n_coarse)
    res = bls.power(periods, duration_grid)
    i_best = np.argmax(res.power)
    p0 = res.period[i_best]
    d0 = res.duration[i_best]
    snr0 = res.depth_snr[i_best]

    # refine around best period
    half = 0.15 * p0  # ±15%
    pmin = max(min_period, p0 - half)
    pmax = min(max_period, p0 + half)
    periods_fine = np.linspace(pmin, pmax, n_fine)
    res2 = bls.power(periods_fine, duration_grid)
    j = np.argmax(res2.power)
    p = float(res2.period[j])
    d = float(res2.duration[j])
    depth = float(res2.depth[j])  # relative depth (fractional)
    snr = float(res2.depth_snr[j])

    # convert outputs
    duration_hours = d * 24.0
    depth_ppm = depth * 1e6

    return p, duration_hours, depth_ppm, snr

def estimate_params(depth_ppm, rstar_rsun=None, teff_k=None, period_days=None, mstar_msun=None):
    """Rough derived quantities: Rp [R_earth], a [AU], a/Rstar (very approximate)."""
    rp_re, a_au, a_over_rstar = None, None, None

    # Rp ~ Rstar * sqrt(depth)
    if rstar_rsun and depth_ppm and depth_ppm > 0:
        depth = depth_ppm / 1e6
        # R_sun to R_earth factor ~ 109.1
        rp_re = float(109.1 * rstar_rsun * np.sqrt(depth))

    # a ~ (P/365)^(2/3) * Mstar^(1/3) AU (Kepler's 3rd; assuming circular)
    if period_days and mstar_msun:
        P_yr = float(period_days) / 365.25
        a_au = float((P_yr ** (2.0/3.0)) * (mstar_msun ** (1.0/3.0)))

    if a_au and rstar_rsun:
        # 1 AU ~ 215 R_sun
        a_over_rstar = float((a_au * 215.032) / rstar_rsun)

    return rp_re, a_au, a_over_rstar

def map_label(idx_or_val):
    """Map class index/value to a string label (supports -1/0/1 and 0/1/2)."""
    v = int(idx_or_val)
    # Common setups we used: [-1,0,1]  or  [0,1,2]
    if v in (-1, 2):  # treat 2 as CONFIRMED in some trainings? We'll do safer:
        # Better mapping: assume [-1,0,1] if negatives appear, else [0,1,2]
        pass
    # Heuristic: if negative present, use [-1,0,1] mapping; else 0=CAND, 1=CONF, 2=FP
    if v == -1:
        return "FALSE POSITIVE"
    if v == 0:
        return "CANDIDATE"
    if v == 1:
        return "CONFIRMED"
    if v == 2:
        # if model used (0=CAND,1=CONF,2=FP)
        return "FALSE POSITIVE"
    return str(v)

# Optional shim if the model was saved as a bare Booster wrapper
try:
    import xgboost as xgb
except Exception:
    xgb = None

class XGBBoosterWrapper:  # keep name for unpickle compatibility
    def predict_proba(self, X):
        if xgb is None:
            raise RuntimeError("xgboost not installed")
        dm = xgb.DMatrix(X)
        return self.booster.predict(dm)
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ----------------------- Main flow -----------------------

def run(csv_path, model_path, out_path=None,
        min_period=0.3, max_period=30.0, n_coarse=1000, n_fine=300):

    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    t_col = find_column(df, POSSIBLE_TIME)
    f_col = find_column(df, POSSIBLE_FLUX)
    if t_col is None or f_col is None:
        raise ValueError(
            f"Could not find time/flux columns. Looked for time in {POSSIBLE_TIME}, "
            f"flux in {POSSIBLE_FLUX}. Your columns: {list(df.columns)}"
        )

    # Extract star metadata if present in CSV (optional)
    star_meta = extract_star_params(df)

    # Prepare light curve
    time = pd.to_numeric(df[t_col], errors="coerce").values
    flux = pd.to_numeric(df[f_col], errors="coerce").values
    time, flux = preprocess_flux(time, flux)

    if len(time) < 50:
        raise ValueError("Not enough valid points after cleaning.")

    # BLS features
    period, dur_h, depth_ppm, snr = bls_features(
        time, flux, min_period=min_period, max_period=max_period,
        n_coarse=n_coarse, n_fine=n_fine
    )

    # Feature vector for model (match what we trained: period, duration, depth, snr)
    X = np.array([[period, dur_h, depth_ppm, snr]], dtype=float)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    # Predict
    try:
        proba = model.predict_proba(X)
    except Exception:
        proba = None
    yhat = model.predict(X)

    label = map_label(int(yhat[0]))

    # Derived (rough) estimates if star params exist
    rp_re, a_au, a_over_rstar = estimate_params(
        depth_ppm=depth_ppm,
        rstar_rsun=star_meta.get("rstar"),
        teff_k=star_meta.get("teff_k"),
        period_days=period,
        mstar_msun=star_meta.get("mstar"),
    )

    # Build output row
    out = {
        "source_path": csv_path,
        "period_days": period,
        "duration_hours": dur_h,
        "depth_ppm": depth_ppm,
        "snr": snr,
        "Pred_Class_Index": int(yhat[0]),
        "Pred_Label": label,
        "Est_Radius_Re": rp_re,
        "Est_SemiMajorAxis_AU": a_au,
        "Est_a_over_Rstar": a_over_rstar,
        "Star_Teff_K": star_meta.get("teff_k"),
        "Star_logg": star_meta.get("logg"),
        "Star_Radius_Rsun": star_meta.get("rstar"),
        "Star_Mass_Msun": star_meta.get("mstar"),
    }

    # Add probabilities if available
    if proba is not None and proba.ndim == 2:
        for j in range(proba.shape[1]):
            out[f"Prob_class_{j}"] = float(proba[0, j])

    # Save CSV
    if out_path is None:
        base, ext = os.path.splitext(csv_path)
        out_path = base + "_prediction.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)

    # Pretty print
    print("✅ Analysis complete")
    print(f"  Period (days):     {period:.6f}")
    print(f"  Duration (hours):  {dur_h:.3f}")
    print(f"  Depth (ppm):       {depth_ppm:.1f}")
    print(f"  SNR:               {snr:.3f}")
    print(f"  Predicted Label:   {label}")
    if rp_re is not None:
        print(f"  Est. Radius (R⊕):  {rp_re:.2f}")
    if a_au is not None:
        print(f"  Est. a (AU):       {a_au:.4f}")
    if a_over_rstar is not None:
        print(f"  Est. a/R★:         {a_over_rstar:.2f}")

    print(f"\n📝 Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze a light-curve CSV and classify planet likelihood.")
    parser.add_argument("--csv", type=str, help="Path to light-curve CSV (contains time & flux columns).")
    parser.add_argument("--model", type=str,
        default=r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Processed\LC_Model_Pro\lightcurve_model.pkl",
        help="Path to trained light-curve model (.pkl).")
    parser.add_argument("--out", type=str, help="Optional output CSV path.")
    parser.add_argument("--min_period", type=float, default=0.3)
    parser.add_argument("--max_period", type=float, default=30.0)
    parser.add_argument("--coarse", type=int, default=1000)
    parser.add_argument("--fine", type=int, default=300)
    args = parser.parse_args()

    csv_path = args.csv
    # Interactive prompt if not provided
    if not csv_path:
        csv_path = input("Paste the path to your light-curve CSV: ").strip().strip('"')

    run(
        csv_path=csv_path,
        model_path=args.model,
        out_path=args.out,
        min_period=args.min_period,
        max_period=args.max_period,
        n_coarse=args.coarse,
        n_fine=args.fine,
    )

if __name__ == "__main__":
    main()
