# ==============================
# Space Invaders — Exoplanet Cumulative CSV -> ML-ready dataset
# No training here. Produces clean feature matrices for XGBoost/LightGBM/Sklearn.
# ==============================

import os
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / ".." / "Data").resolve()
OUT_DIR  = (BASE_DIR / ".." / ".." / "Processed").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Find the newest cumulative_*.csv ----------
candidates = sorted(DATA_DIR.glob("cumulative_*.csv"), key=os.path.getmtime, reverse=True)
if not candidates:
    raise FileNotFoundError(f"No cumulative_*.csv found in {DATA_DIR}")
csv_path = candidates[0]
print(f"Using CSV: {csv_path.name}")

# ---------- Load: NASA files have comment lines starting with '#' ----------
# Pandas will ignore lines starting with '#', and use the first non-# line as header.
df = pd.read_csv(
    csv_path,
    comment="#",
    encoding="utf-8-sig",
    engine="c",        # fast and fine for comma-separated cumulative files
    low_memory=False   # ok with C engine
)

# Drop unnamed junk columns if any
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

print("Initial shape after comment-skip:", df.shape)

# ---------- Choose label & candidate feature columns ----------
# Common columns in NASA KOI cumulative table (some may be missing depending on release).
# We request a broad set; we'll only keep those that exist.
label_col = "koi_disposition"  # CONFIRMED / CANDIDATE / FALSE POSITIVE

numeric_candidates = [
    "koi_period",       # orbital period (days)
    "koi_duration",     # transit duration (hrs)
    "koi_depth",        # transit depth (ppm)
    "koi_impact",       # impact parameter
    "koi_prad",         # planet radius (Earth radii)
    "koi_sma",          # semi-major axis (AU)
    "koi_teq",          # equilibrium temperature (K)
    "koi_insol",        # incident flux (Earth=1)
    "koi_model_snr",    # model SNR
]

stellar_numeric = [
    "koi_steff",        # stellar effective temp (K)
    "koi_slogg",        # stellar logg (cgs)
    "koi_smet",         # stellar metallicity [Fe/H]
    "koi_srad",         # stellar radius (Solar radii)
    "koi_smass",        # stellar mass (Solar masses) — sometimes present
]

boolean_flag_cols = [
    "koi_fpflag_nt",    # not transit-like
    "koi_fpflag_ss",    # stellar eclipse
    "koi_fpflag_co",    # centroid offset
    "koi_fpflag_ec"     # ephemeris match indicates contaminant
]

id_like = ["kepid", "kepoi_name", "kepler_name"]  # useful to keep for joins/analysis, not for model

# Keep only columns that actually exist
present_numeric = [c for c in (numeric_candidates + stellar_numeric) if c in df.columns]
present_flags   = [c for c in boolean_flag_cols if c in df.columns]
present_ids     = [c for c in id_like if c in df.columns]

needed_cols = [label_col] + present_numeric + present_flags + present_ids
missing_label = label_col not in df.columns
df = df[[c for c in needed_cols if c in df.columns]].copy()

print("Columns selected:", list(df.columns))
if missing_label:
    print("⚠️ Warning: label column 'koi_disposition' not found. We'll still prep features only.")

# ---------- Basic cleaning ----------
# Coerce numeric columns (errors='coerce' turns bad strings -> NaN)
for c in present_numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Flags should be 0/1
for c in present_flags:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# Drop rows where EVERY feature is NaN (useless rows)
feature_cols = present_numeric + present_flags
df = df.dropna(axis=0, how="all", subset=feature_cols)

# Drop columns that are >40% missing (too sparse)
missing_frac = df[feature_cols].isna().mean()
keep_cols = [c for c in feature_cols if missing_frac.get(c, 0) <= 0.40]
dropped_cols = sorted(set(feature_cols) - set(keep_cols))
if dropped_cols:
    print("Dropping sparse feature columns (>40% NaN):", dropped_cols)

feature_cols = keep_cols

# Impute remaining numeric with median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df[feature_cols])

# Build y (if label exists)
y = None
label_map = None
if not missing_label:
    # Keep full string labels; tree models are fine with numeric y, but we export both.
    y = df[label_col].astype(str).values
    # Make a mapping to integers (useful later)
    classes = sorted(pd.Series(y).unique().tolist())
    label_map = {cls: i for i, cls in enumerate(classes)}
else:
    print("No label present; exporting features only.")

# ---------- Train/Test split (no training here, just to hand teams ready splits) ----------
if y is not None:
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=42, stratify=y
    )
else:
    # If no label, just split features
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, df.index.values, test_size=0.2, random_state=42
    )
    y_train = y_test = None

# ---------- Save outputs ----------
np.save(OUT_DIR / "X_train.npy", X_train)
np.save(OUT_DIR / "X_test.npy",  X_test)
if y_train is not None:
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_test.npy",  y_test)

# Helpful CSVs for inspection / non-numpy users
pd.DataFrame(X_train, columns=feature_cols).to_csv(OUT_DIR / "X_train.csv", index=False)
pd.DataFrame(X_test,  columns=feature_cols).to_csv(OUT_DIR / "X_test.csv",  index=False)
if y_train is not None:
    pd.Series(y_train, name=label_col).to_csv(OUT_DIR / "y_train.csv", index=False)
    pd.Series(y_test,  name=label_col).to_csv(OUT_DIR / "y_test.csv",  index=False)

# Metadata to re-use later (feature names, imputers, mappings)
meta = {
    "source_csv": str(csv_path.name),
    "feature_cols": feature_cols,
    "dropped_sparse_cols": dropped_cols,
    "id_columns_in_original": present_ids,
    "label_column": label_col if not missing_label else None,
    "label_map": label_map,               # {"CANDIDATE": 0, "CONFIRMED": 1, ...}
    "imputer": "SimpleImputer(median)",   # re-create at inference or pickle it if desired
    "split": {"test_size": 0.2, "random_state": 42, "stratify": bool(y is not None)},
}
with open(OUT_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("✅ Done.")
print(f"Saved to: {OUT_DIR}")
print(f"Features used ({len(feature_cols)}): {feature_cols}")
if y_train is not None:
    print(f"Label classes: {list(label_map.keys())}")
