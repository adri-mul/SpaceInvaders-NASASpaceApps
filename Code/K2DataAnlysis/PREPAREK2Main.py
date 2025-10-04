# ============================================
# Space Invaders — TOI & K2 datasets -> ML-ready
# ============================================
import os
import csv
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / ".." / "Data").resolve()
OUT_DIR  = (BASE_DIR / ".." / ".." / "Processed").resolve()

# Filenames you provided (can exist alongside others)
TOI_FILE   = "TOI_2025.10.04_09.07.03.csv"
K2PC_FILE  = "k2pandc_2025.10.04_09.07.07.csv"

# -----------------------------
# Robust CSV loader (no edits)
# -----------------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    """Load a NASA/MAST-style CSV with metadata lines and uncertain delimiter."""
    # Try to sniff delimiter
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
        print(f"[{path.name}] Detected delimiter: {repr(delimiter)}")
    except Exception as e:
        print(f"[{path.name}] ⚠️ Could not auto-detect delimiter ({e}), defaulting to comma.")
        delimiter = ","

    # Try fast C engine first
    try:
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            engine="c",
            comment="#",
            encoding="utf-8-sig",
        )
        return df
    except Exception as e:
        print(f"[{path.name}] ⚠️ Fallback to python engine due to: {e}")
        df = pd.read_csv(
            path,
            sep=None,           # let pandas guess
            engine="python",    # more tolerant
            comment="#",
            encoding="utf-8-sig",
            on_bad_lines="skip"
        )
        return df

# ------------------------------------------
# Column alias mapping (TOI & K2 variations)
# ------------------------------------------
# We map many possible column names (as they vary by export) to a canonical name.
ALIASES: Dict[str, List[str]] = {
    # Target/label
    "label": [
        "Disposition", "tfopwg_disp", "koi_disposition", "disposition",
        "k2_disposition", "status"
    ],

    # IDs / names (kept for reference only)
    "id_star": ["TIC ID", "tic_id", "kepid", "epic_id", "epic", "star_id"],
    "id_object": ["TOI", "toi", "epic_candname", "epic_name", "k2_name", "kepoi_name", "kepler_name"],

    # Orbital & transit
    "period_days": ["Period (days)", "period", "pl_orbper", "koi_period", "orbital_period_days"],
    "duration_hours": ["Duration (hours)", "duration_hours", "koi_duration", "pl_trandurh"],
    "depth_ppm": ["Depth (ppm)", "depth_ppm", "koi_depth", "pl_trandep"],

    # Planet properties
    "prad_re": ["Planet Radius (R_Earth)", "koi_prad", "pl_rade", "planet_radius_rearth"],

    # Stellar parameters
    "teff_k": ["Stellar Teff (K)", "st_teff", "koi_steff", "teff", "stellar_teff_k"],
    "logg_cgs": ["Stellar log g (cgs)", "st_logg", "koi_slogg", "logg", "stellar_logg_cgs"],
    "feh": ["Stellar [Fe/H] (dex)", "st_met", "koi_smet", "feh", "stellar_feh"],
    "srad_rsun": ["Stellar Radius (R_Sun)", "st_rad", "koi_srad", "stellar_radius_rsun"],
    "smass_msun": ["Stellar Mass (M_Sun)", "st_mass", "koi_smass", "stellar_mass_msun"],

    # Flags (disposition clues)
    "fpflag_nt": ["koi_fpflag_nt"],
    "fpflag_ss": ["koi_fpflag_ss"],
    "fpflag_co": ["koi_fpflag_co"],
    "fpflag_ec": ["koi_fpflag_ec"],

    # Misc strength/SNR
    "model_snr": ["koi_model_snr", "model_snr"],
    "insol": ["koi_insol", "insolation_flux"],
    "teq_k": ["koi_teq", "equilibrium_temp_k"],
}

# Which canonical features we’ll keep as numeric inputs (order matters for consistent matrices)
FEATURE_ORDER = [
    "period_days", "duration_hours", "depth_ppm",
    "prad_re", "insol", "teq_k", "model_snr",
    "teff_k", "logg_cgs", "feh", "srad_rsun", "smass_msun",
    "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"
]

# Reference cols to carry through (not used for ML features)
REF_KEEP = ["id_star", "id_object"]

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with canonical column names (if found)."""
    out = pd.DataFrame(index=df.index)
    # Map label first (if present)
    for canonical, aliases in ALIASES.items():
        for name in aliases:
            if name in df.columns:
                out[canonical] = df[name]
                break  # take first match

    # Keep only the columns we care about
    keep = ["label"] + FEATURE_ORDER + REF_KEEP
    cols_present = [c for c in keep if c in out.columns]
    return out[cols_present].copy()

def coerce_and_impute(df_mapped: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], Dict]:
    """Coerce numerics, cast flags, impute, return (X, y, feature_names, meta_info)."""
    # Build feature list from what’s present
    feature_cols = [c for c in FEATURE_ORDER if c in df_mapped.columns]

    # Coerce numerics
    for c in feature_cols:
        df_mapped[c] = pd.to_numeric(df_mapped[c], errors="coerce")

    # Flags to 0/1 if present
    for flag in ["fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"]:
        if flag in df_mapped.columns:
            df_mapped[flag] = pd.to_numeric(df_mapped[flag], errors="coerce").fillna(0).astype(int)

    # Drop rows where *all* feature values are NaN
    df_work = df_mapped.dropna(axis=0, how="all", subset=feature_cols).copy()

    # Column missing-rate filter (drop features with >40% NaN)
    missing_frac = df_work[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missing_frac.get(c, 0) <= 0.40]
    dropped_sparse = sorted(set(FEATURE_ORDER).intersection(df_mapped.columns) - set(feature_cols))

    # Impute remaining NaNs
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df_work[feature_cols])

    # Optional label
    y = None
    label_map = None
    if "label" in df_work.columns:
        y_series = df_work["label"].astype(str)
        classes = sorted(y_series.dropna().unique().tolist())
        label_map = {cls: i for i, cls in enumerate(classes)}
        y = y_series.map(label_map).values

    meta = {
        "feature_cols": feature_cols,
        "dropped_sparse_cols": dropped_sparse,
        "label_map": label_map,
        "ref_columns_present": [c for c in REF_KEEP if c in df_mapped.columns],
    }
    return X, y, feature_cols, meta

def export_artifacts(X, y, feature_cols, meta, out_dir: Path, dataset_name: str):
    out_dir = out_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train/test split (no training here)
    if y is not None and len(np.unique(y[~pd.isna(y)])) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, np.zeros(X.shape[0]), test_size=0.2, random_state=42
        )
        y_train = y_test = None

    # Numpy
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy",  X_test)
    if y_train is not None:
        np.save(out_dir / "y_train.npy", y_train)
        np.save(out_dir / "y_test.npy",  y_test)

    # CSV mirrors (for inspection)
    pd.DataFrame(X_train, columns=feature_cols).to_csv(out_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test,  columns=feature_cols).to_csv(out_dir / "X_test.csv",  index=False)
    if y_train is not None:
        pd.Series(y_train, name="label").to_csv(out_dir / "y_train.csv", index=False)
        pd.Series(y_test,  name="label").to_csv(out_dir / "y_test.csv",  index=False)

    # Metadata
    meta_payload = {
        "feature_cols": feature_cols,
        "label_map": meta["label_map"],
        "dropped_sparse_cols": meta["dropped_sparse_cols"],
        "ref_columns_present": meta["ref_columns_present"],
        "dataset_name": dataset_name,
    }
    with open(out_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"✅ Saved ML-ready artifacts to: {out_dir}")
    if meta["label_map"]:
        print(f"Label classes: {list(meta['label_map'].keys())}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

def process_one(filename: str, dataset_name: str):
    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV not found: {csv_path}")

    # Load
    raw = read_csv_robust(csv_path)
    # Drop unnamed junk cols
    raw = raw.loc[:, ~raw.columns.str.match(r"^Unnamed")]

    # Map to canonical columns
    mapped = map_columns(raw)
    print(f"[{dataset_name}] Columns present: {list(mapped.columns)}")

    # Coerce + impute -> matrices
    X, y, feature_cols, meta = coerce_and_impute(mapped)

    # Export
    export_artifacts(X, y, feature_cols, meta, OUT_DIR, dataset_name)

def main():
    print("=== Processing TOI dataset ===")
    process_one(TOI_FILE, dataset_name="TOI")

    print("\n=== Processing K2 P&C dataset ===")
    process_one(K2PC_FILE, dataset_name="K2")

if __name__ == "__main__":
    main()
    