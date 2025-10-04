# ============================================
# Space Invaders — TOI (TESS Objects of Interest) -> ML-ready
# ============================================
import os, csv, json, glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / ".." / "Data").resolve()
OUT_DIR  = (BASE_DIR / ".." / ".." / "Processed" / "TOI").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If you want to pin a specific file, set this; else newest TOI_*.csv will be used.
PINNED_FILE = None  # e.g., "TOI_2025.10.04_09.07.03.csv"

# ---- Robust CSV loader ----
def read_csv_robust(path: Path) -> pd.DataFrame:
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
        return pd.read_csv(path, delimiter=delimiter, engine="c", comment="#", encoding="utf-8-sig")
    except Exception as e:
        print(f"[{path.name}] ⚠️ Fallback to python engine due to: {e}")
        return pd.read_csv(path, sep=None, engine="python", comment="#", encoding="utf-8-sig", on_bad_lines="skip")

# ---- Column aliases for TOI ----
ALIASES: Dict[str, List[str]] = {
    # Label (TOI often uses TESS vetting fields)
    "label": ["Disposition", "tfopwg_disp", "disposition", "status"],

    # IDs
    "id_star": ["TIC ID", "tic_id", "ticid", "star_id"],
    "id_object": ["TOI", "toi", "toi_name", "object_name"],

    # Orbital / transit
    "period_days": ["Period (days)", "period", "pl_orbper", "koi_period"],
    "duration_hours": ["Duration (hours)", "duration_hours", "pl_trandurh", "koi_duration"],
    "depth_ppm": ["Depth (ppm)", "depth_ppm", "pl_trandep", "koi_depth"],

    # Planet properties
    "prad_re": ["Planet Radius (R_Earth)", "pl_rade", "koi_prad", "planet_radius_rearth"],

    # Stellar params
    "teff_k": ["Stellar Teff (K)", "st_teff", "teff", "stellar_teff_k"],
    "logg_cgs": ["Stellar log g (cgs)", "st_logg", "logg", "stellar_logg_cgs"],
    "feh": ["Stellar [Fe/H] (dex)", "st_met", "feh", "stellar_feh"],
    "srad_rsun": ["Stellar Radius (R_Sun)", "st_rad", "stellar_radius_rsun"],
    "smass_msun": ["Stellar Mass (M_Sun)", "st_mass", "stellar_mass_msun"],

    # Strength / SNR (if present)
    "model_snr": ["model_snr", "snr", "koi_model_snr"],

    # Flags (rare in TOI exports; mapped if present)
    "fpflag_nt": ["koi_fpflag_nt"],
    "fpflag_ss": ["koi_fpflag_ss"],
    "fpflag_co": ["koi_fpflag_co"],
    "fpflag_ec": ["koi_fpflag_ec"],

    # Insolation / Teq if included
    "insol": ["insolation_flux", "koi_insol"],
    "teq_k": ["equilibrium_temp_k", "koi_teq"],
}

FEATURE_ORDER = [
    "period_days", "duration_hours", "depth_ppm",
    "prad_re", "insol", "teq_k", "model_snr",
    "teff_k", "logg_cgs", "feh", "srad_rsun", "smass_msun",
    "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"
]

REF_KEEP = ["id_star", "id_object"]

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in ALIASES.items():
        for name in aliases:
            if name in df.columns:
                out[canonical] = df[name]
                break
    keep = ["label"] + FEATURE_ORDER + REF_KEEP
    return out[[c for c in keep if c in out.columns]].copy()

def coerce_impute(df_mapped: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], dict]:
    # Establish available features
    feature_cols = [c for c in FEATURE_ORDER if c in df_mapped.columns]

    # Coerce numerics
    for c in feature_cols:
        df_mapped[c] = pd.to_numeric(df_mapped[c], errors="coerce")

    # Flags to 0/1
    for flag in ["fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"]:
        if flag in df_mapped.columns:
            df_mapped[flag] = pd.to_numeric(df_mapped[flag], errors="coerce").fillna(0).astype(int)

    # Drop rows with *all* NaN features
    df_work = df_mapped.dropna(axis=0, how="all", subset=feature_cols).copy()

    # Drop features with >40% NaN
    miss = df_work[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if miss.get(c, 0) <= 0.40]
    dropped_sparse = sorted(set(FEATURE_ORDER).intersection(df_mapped.columns) - set(feature_cols))

    # Impute remaining NaNs
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df_work[feature_cols])

    # Optional label
    y = None
    label_map = None
    if "label" in df_work.columns:
        y_ser = df_work["label"].astype(str)
        classes = sorted(y_ser.dropna().unique().tolist())
        if len(classes) >= 2:
            label_map = {cls: i for i, cls in enumerate(classes)}
            y = y_ser.map(label_map).values

    meta = {
        "feature_cols": feature_cols,
        "dropped_sparse_cols": dropped_sparse,
        "label_map": label_map,
        "ref_columns_present": [c for c in REF_KEEP if c in df_mapped.columns],
    }
    return X, y, feature_cols, meta

def export(X, y, feature_cols, meta):
    # Split (no training)
    if y is not None and len(np.unique(y[~pd.isna(y)])) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train = y_test = None

    # Save
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "X_test.npy",  X_test)
    pd.DataFrame(X_train, columns=feature_cols).to_csv(OUT_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test,  columns=feature_cols).to_csv(OUT_DIR / "X_test.csv",  index=False)
    if y_train is not None:
        np.save(OUT_DIR / "y_train.npy", y_train)
        np.save(OUT_DIR / "y_test.npy",  y_test)
        pd.Series(y_train, name="label").to_csv(OUT_DIR / "y_train.csv", index=False)
        pd.Series(y_test,  name="label").to_csv(OUT_DIR / "y_test.csv",  index=False)

    meta_payload = {
        "feature_cols": feature_cols,
        "label_map": meta["label_map"],
        "dropped_sparse_cols": meta["dropped_sparse_cols"],
        "ref_columns_present": meta["ref_columns_present"],
        "dataset_name": "TOI",
    }
    with open(OUT_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"✅ Saved ML-ready artifacts to: {OUT_DIR}")
    if meta["label_map"]:
        print(f"Label classes: {list(meta['label_map'].keys())}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

def main():
    if PINNED_FILE:
        toi_path = DATA_DIR / PINNED_FILE
        if not toi_path.exists():
            raise FileNotFoundError(f"❌ CSV not found: {toi_path}")
    else:
        # Pick newest TOI_*.csv
        matches = sorted(DATA_DIR.glob("TOI_*.csv"), key=os.path.getmtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No TOI_*.csv found in {DATA_DIR}")
        toi_path = matches[0]

    print(f"Using TOI file: {toi_path.name}")
    raw = read_csv_robust(toi_path)
    raw = raw.loc[:, ~raw.columns.str.match(r"^Unnamed")]  # drop unnamed junk if present

    mapped = map_columns(raw)
    print("Columns mapped:", list(mapped.columns))

    X, y, feature_cols, meta = coerce_impute(mapped)
    export(X, y, feature_cols, meta)

if __name__ == "__main__":
    main()
