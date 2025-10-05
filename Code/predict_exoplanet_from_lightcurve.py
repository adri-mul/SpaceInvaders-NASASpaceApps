# ===============================================================
# Space Invaders — Predict from Light Curve (w/ built-in fetching)
# ===============================================================
# Examples (PowerShell):
# 1) Demo (auto downloads a few known targets):
#    python Code\predict_exoplanet_from_lightcurve.py `
#      --model "C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Processed\LC_Model_Pro\lightcurve_model.pkl" `
#      --demo `
#      --out "C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Predictions\demo_preds.csv"
#
# 2) Generic identifiers (TIC/KIC or star name if resolvable):
#    python Code\predict_exoplanet_from_lightcurve.py `
#      --model "C:\...\lightcurve_model.pkl" `
#      --ids "TIC 150428135" "KIC 11446443" `
#      --mission tess `
#      --out "C:\...\preds_from_ids.csv"
#
# 3) Raw CSV(s) with columns time,flux:
#    python Code\predict_exoplanet_from_lightcurve.py `
#      --model "C:\...\lightcurve_model.pkl" `
#      --csv "C:\path\lc1.csv" "C:\path\lc2.csv" `
#      --out "C:\...\preds_from_csv.csv"
#
# Supports your combined/tabular XGB model too if it expects [period, duration, depth, snr].
# ===============================================================

import argparse, os, math, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

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

AU_PER_RSUN = 1.0 / 215.032  # 1 AU ≈ 215.032 R_sun
RE_PER_RSUN = 109.1          # R_sun ≈ 109.1 R_earth


# ---------- Light-curve Readers ----------
def read_lightcurve_csv(path: Path):
    df = pd.read_csv(path)
    tcol = next((c for c in ["time","Time","TIME","btjd","bjd","jd","t"] if c in df.columns), None)
    fcol = next((c for c in ["flux","Flux","FLUX","pdcsap_flux","sap_flux","f"] if c in df.columns), None)
    if not tcol or not fcol:
        raise ValueError(f"{path}: CSV must contain time and flux columns (e.g., time,flux).")
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    f = pd.to_numeric(df[fcol], errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    med = np.nanmedian(f)
    if not np.isfinite(med) or med == 0:
        raise ValueError(f"{path}: Flux median invalid.")
    f = f / med
    return t, f


# ---------- Unified Lightkurve fetcher ----------
def fetch_lightcurve(identifier: str, mission="tess"):
    """
    Identifier can be:
      - 'TIC 123456789'  or just '123456789' with mission='tess'
      - 'KIC 11446443'   or a resolvable object name
    """
    if not HAS_LK:
        raise RuntimeError("lightkurve not installed. pip install 'lightkurve[all]'")
    ident = identifier.strip()
    # Heuristics to keep it flexible:
    if ident.upper().startswith(("TIC", "KIC", "EPIC")):
        query = ident
    else:
        # If looks numeric and mission is tess, assume TIC
        if ident.isdigit() and mission.lower().startswith("tess"):
            query = f"TIC {ident}"
        elif ident.isdigit():
            query = ident  # let lightkurve resolve
        else:
            query = ident  # try name resolve
    srch = lk.search_lightcurve(query, mission=mission.upper())
    if len(srch) == 0:
        raise FileNotFoundError(f"No light curve found for '{identifier}' ({mission}).")
    lc = srch.download().remove_nans()
    try:
        lc = lc.flatten(window_length=401).normalize()
    except Exception:
        lc = lc.normalize()
    return lc.time.value, lc.flux.value


# ---------- BLS features ----------
def bls_features(time, flux, min_period=0.5, max_period=20.0,
                 n_periods=1200, duration_frac=0.05):
    if not HAS_AST:
        raise RuntimeError("astropy is required for BLS (pip install astropy).")
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if len(t) < 60:
        raise ValueError("Not enough points for BLS (need ~60+).")
    f = f / np.nanmedian(f)
    periods = np.linspace(min_period, max_period, int(n_periods))
    bls = BoxLeastSquares(t, f)
    res = bls.power(periods, duration_frac)
    i = int(np.nanargmax(res.power))
    period = float(res.period[i])
    duration_h = float(res.duration[i] * 24.0)
    depth_ppm = float(max(res.depth[i], 0.0) * 1e6)
    snr = float(getattr(res, "depth_snr", res.power)[i])
    return period, duration_h, depth_ppm, snr


# ---------- Stellar params from TIC ----------
def fetch_tic_params_from_any(identifier: str):
    """Return dict with teff, logg, rad (R_sun), mass (M_sun) if available via TIC."""
    out = {"teff": None, "logg": None, "rad": None, "mass": None, "tic_id": None}
    if not HAS_MAST:
        return out
    try:
        # Try to extract TIC number if present
        tic_guess = None
        txt = identifier.upper()
        if "TIC" in txt:
            digits = "".join(ch for ch in txt if ch.isdigit())
            if digits.isdigit():
                tic_guess = digits
        # Query TIC by object if we can’t parse a number
        query_str = f"TIC {tic_guess}" if tic_guess else identifier
        res = Catalogs.query_object(query_str, catalog="TIC")
        if len(res) == 0:
            return out
        row = res[0]
        def _get(field):
            try:
                v = row[field]
                return float(v) if v is not None else None
            except Exception:
                return None
        out["teff"] = _get("Teff")
        out["logg"] = _get("logg")
        out["rad"]  = _get("rad")
        out["mass"] = _get("mass")
        # Get TIC ID field if present
        try:
            out["tic_id"] = int(row["ID"])
        except Exception:
            out["tic_id"] = tic_guess
        return out
    except Exception:
        return out


# ---------- Estimators ----------
def estimate_radius_earth(depth_ppm: float, stellar_radius_rsun: float | None):
    """Rp ≈ sqrt(depth) * R_star; depth in ppm, R_star in R_sun → output in R_earth."""
    if stellar_radius_rsun is None or stellar_radius_rsun <= 0 or not np.isfinite(stellar_radius_rsun):
        return None
    depth_frac = max(depth_ppm, 0.0) / 1e6
    rp_rsun = math.sqrt(max(depth_frac, 0.0)) * stellar_radius_rsun
    rp_re = rp_rsun * RE_PER_RSUN
    return rp_re

def estimate_semi_major_axis(period_days: float, stellar_mass_msun: float | None, stellar_radius_rsun: float | None):
    """
    a(AU) ≈ (P_years^2 * M_star)^(1/3) in solar units.
    Also return a/R_star if R_star available.
    """
    if stellar_mass_msun is None or stellar_mass_msun <= 0 or not np.isfinite(stellar_mass_msun):
        # crude fallback: assume M ≈ R for MS stars if radius known
        if stellar_radius_rsun is not None and stellar_radius_rsun > 0:
            stellar_mass_msun = stellar_radius_rsun
        else:
            return None, None
    P_years = period_days / 365.25
    a_AU = (P_years**2 * stellar_mass_msun) ** (1.0/3.0)
    a_over_Rs = None
    if stellar_radius_rsun is not None and stellar_radius_rsun > 0:
        a_over_Rs = a_AU / (stellar_radius_rsun * AU_PER_RSUN)  # AU → R_sun units
    return a_AU, a_over_Rs


# ---------- Model loader & predictor ----------
def load_model(model_path: str):
    obj = joblib.load(model_path)
    # could be plain model, or dict {"model":..., "classes":...}
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        classes = obj.get("classes", None)
    else:
        model = obj
        classes = getattr(model, "classes_", None)
    return model, classes

def predict_classes(model, X):
    proba = None
    yhat = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    if hasattr(model, "predict"):
        yhat = model.predict(X)
    # handle xgboost Booster-like
    try:
        import xgboost as xgb
        if not hasattr(model, "predict") and hasattr(model, "booster"):
            dm = xgb.DMatrix(X)
            proba = model.booster.predict(dm)
            yhat = np.argmax(proba, axis=1)
        elif model.__class__.__name__ == "Booster":
            dm = xgb.DMatrix(X)
            proba = model.predict(dm)
            yhat = np.argmax(proba, axis=1)
    except Exception:
        pass
    return yhat, proba

DEFAULT_LABELS = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

def map_label_indices(yhat, classes_attr=None, provided_map_json=None):
    if provided_map_json:
        mapping = json.loads(Path(provided_map_json).read_text(encoding="utf-8"))
        idx_to_name = {int(k): v for k, v in mapping.items()}
        return [idx_to_name.get(int(i), str(i)) for i in yhat]
    if classes_attr is not None:
        unique_vals = list(classes_attr)
        idx_to_val = {i: int(v) for i, v in enumerate(unique_vals)}
        def name_for(idx):
            v = idx_to_val.get(int(idx), idx)
            if v <= -1: return "FALSE POSITIVE"
            if v == 0:  return "CANDIDATE"
            return "CONFIRMED"
        return [name_for(i) for i in yhat]
    return [DEFAULT_LABELS.get(int(i), str(i)) for i in yhat]


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Predict exoplanet classification and estimate size/orbit from light curves.")
    ap.add_argument("--model", required=True, help="Path to model.pkl (tabular or lightcurve model).")
    ap.add_argument("--csv", nargs="*", help="One or more CSV light-curves (time,flux).")
    ap.add_argument("--tic_ids", nargs="*", help="TIC IDs to download automatically.")
    ap.add_argument("--ids", nargs="*", help="Generic identifiers (e.g., 'TIC 150428135', 'KIC 11446443', star name).")
    ap.add_argument("--mission", default="tess", help="Mission for downloads: tess or kepler.")
    ap.add_argument("--label_map_json", default=None, help='Optional JSON index->label mapping.')
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--min_period", type=float, default=0.5)
    ap.add_argument("--max_period", type=float, default=20.0)
    ap.add_argument("--demo", action="store_true", help="Fetch a small demo set if no inputs given.")
    args = ap.parse_args()

    model, classes_attr = load_model(args.model)

    # Auto demo if requested or no inputs provided
    targets = []
    if args.csv or args.tic_ids or args.ids:
        pass
    elif args.demo:
        # Curated small set (fast). You can change these:
        targets = ["TIC 150428135", "TIC 307210830", "KIC 11446443"]
        print(f"🧪 DEMO MODE: using {targets}")
    else:
        raise SystemExit("Provide --csv, --tic_ids, --ids, or --demo.")

    rows = []

    # 1) CSV inputs
    if args.csv:
        for p in args.csv:
            try:
                t, f = read_lightcurve_csv(Path(p))
                P, dur_h, depth_ppm, snr = bls_features(t, f, args.min_period, args.max_period)
                name = Path(p).stem
                # Try stellar params from filename token 'TIC ####'
                star = fetch_tic_params_from_any(name) if HAS_MAST else {"teff":None,"logg":None,"rad":None,"mass":None,"tic_id":None}
                rows.append({
                    "source": name,
                    "origin": "csv",
                    "identifier": name,
                    "period_days": P,
                    "duration_hours": dur_h,
                    "depth_ppm": depth_ppm,
                    "snr": snr,
                    "teff_k": star["teff"],
                    "logg_cgs": star["logg"],
                    "star_rad_rsun": star["rad"],
                    "star_mass_msun": star["mass"],
                })
            except Exception as e:
                print(f"⚠️ Skipping {p}: {e}")

    # 2) TIC IDs
    if args.tic_ids:
        for tid in args.tic_ids:
            ident = f"TIC {tid}"
            targets.append(ident)

    # 3) Generic IDs
    if args.ids:
        targets.extend(args.ids)

    # 4) Fetch all targets via Lightkurve
    for ident in targets:
        try:
            t, f = fetch_lightcurve(ident, mission=args.mission)
            P, dur_h, depth_ppm, snr = bls_features(t, f, args.min_period, args.max_period)
            star = fetch_tic_params_from_any(ident) if HAS_MAST else {"teff":None,"logg":None,"rad":None,"mass":None,"tic_id":None}
            rows.append({
                "source": ident.replace(" ", ""),
                "origin": "download",
                "identifier": ident,
                "period_days": P,
                "duration_hours": dur_h,
                "depth_ppm": depth_ppm,
                "snr": snr,
                "teff_k": star["teff"],
                "logg_cgs": star["logg"],
                "star_rad_rsun": star["rad"],
                "star_mass_msun": star["mass"],
            })
        except Exception as e:
            print(f"⚠️ Skipping {ident}: {e}")

    if not rows:
        raise SystemExit("No valid light curves to score.")

    df = pd.DataFrame(rows)

    # Features for your model (order matters)
    X = df[["period_days", "duration_hours", "depth_ppm", "snr"]].copy().to_numpy()

    yhat, proba = predict_classes(model, X)
    if yhat is None:
        raise RuntimeError("Model could not produce predictions.")

    human = map_label_indices(yhat, classes_attr=classes_attr, provided_map_json=args.label_map_json)

    # Estimates
    est_radius_re = []
    est_a_AU = []
    est_a_over_Rs = []
    rel_period_earth = []
    for _, r in df.iterrows():
        rp = estimate_radius_earth(r["depth_ppm"], r["star_rad_rsun"])
        a_AU, a_over_Rs = estimate_semi_major_axis(r["period_days"], r["star_mass_msun"], r["star_rad_rsun"])
        est_radius_re.append(None if rp is None else float(rp))
        est_a_AU.append(None if a_AU is None else float(a_AU))
        est_a_over_Rs.append(None if a_over_Rs is None else float(a_over_Rs))
        rel_period_earth.append(float(r["period_days"] / 365.25))

    out_df = df.copy()
    out_df["Pred_Class_Index"] = yhat
    out_df["Pred_Label"] = human
    if proba is not None:
        for j in range(proba.shape[1]):
            out_df[f"Prob_class_{j}"] = proba[:, j]
    out_df["Est_Radius_Re"] = est_radius_re
    out_df["Est_SemiMajorAxis_AU"] = est_a_AU
    out_df["Est_a_over_Rstar"] = est_a_over_Rs
    out_df["Period_vs_Earth"] = rel_period_earth

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved predictions → {out_path}")
    print("First few rows:\n", out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
