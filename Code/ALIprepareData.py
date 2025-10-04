# ============================================
# Space Invaders — Combine KOI Cumulative + TOI + K2 -> ML-ready
# ============================================
import os, csv, json, glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "Data").resolve()           # adjust if your layout differs
OUT_DIR  = (BASE_DIR / ".." / "Processed" / "Combined").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Patterns (newest file for each) -------------
PATTERNS = {
    "KOI": "cumulative_*.csv",
    "TOI": "TOI_*.csv",
    "K2" : "k2pandc_*.csv",
}

# ------- Canonical aliases (union over all three sources) -------
ALIASES: Dict[str, List[str]] = {
    # Label / disposition
    "label": [
        "koi_disposition", "Disposition", "tfopwg_disp", "disposition", "k2_disposition", "status"
    ],

    # IDs to carry (not features)
    "id_star": ["kepid", "TIC ID", "tic_id", "ticid", "epic_id", "epic", "star_id"],
    "id_object": ["kepoi_name", "kepler_name", "TOI", "toi", "toi_name", "epic_candname", "k2_name", "object_name"],

    # Orbital / transit
    "period_days": ["koi_period", "Period (days)", "period", "pl_orbper", "orbital_period_days"],
    "duration_hours": ["koi_duration", "Duration (hours)", "duration_hours", "pl_trandurh"],
    "depth_ppm": ["koi_depth", "Depth (ppm)", "depth_ppm", "pl_trandep"],

    # Planet
    "prad_re": ["koi_prad", "Planet Radius (R_Earth)", "pl_rade", "planet_radius_rearth"],

    # Stellar
    "teff_k": ["koi_steff", "Stellar Teff (K)", "st_teff", "teff", "stellar_teff_k"],
    "logg_cgs": ["koi_slogg", "Stellar log g (cgs)", "st_logg", "logg", "stellar_logg_cgs"],
    "feh": ["koi_smet", "Stellar [Fe/H] (dex)", "st_met", "feh", "stellar_feh"],
    "srad_rsun": ["koi_srad", "Stellar Radius (R_Sun)", "st_rad", "stellar_radius_rsun"],
    "smass_msun": ["koi_smass", "Stellar Mass (M_Sun)", "st_mass", "stellar_mass_msun"],

    # Additional numeric
    "model_snr": ["koi_model_snr", "model_snr", "snr"],
    "insol": ["koi_insol", "insolation_flux"],
    "teq_k": ["koi_teq", "equilibrium_temp_k"],

    # Flags -> numeric binary
    "fpflag_nt": ["koi_fpflag_nt"],
    "fpflag_ss": ["koi_fpflag_ss"],
    "fpflag_co": ["koi_fpflag_co"],
    "fpflag_ec": ["koi_fpflag_ec"],
}

# Order your features consistently
FEATURE_ORDER = [
    "period_days", "duration_hours", "depth_ppm",
    "prad_re", "insol", "teq_k", "model_snr",
    "teff_k", "logg_cgs", "feh", "srad_rsun", "smass_msun",
    "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"
]

# Keep for reference (not used as features)
REF_KEEP = ["id_star", "id_object", "source"]


# ----------------- Robust CSV loader -----------------
def _detect_delimiter_skip_comments(path: Path) -> str:
    """
    Detect delimiter from NON-comment lines only (ignores lines starting with '#').
    Fallback to common delimiters if sniff fails.
    """
    candidates = [",", "\t", ";", "|", " "]
    non_comment = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for _ in range(3000):  # scan some lines
                line = f.readline()
                if not line:
                    break
                if not line.lstrip().startswith("#") and line.strip():
                    non_comment.append(line)
                if len(non_comment) >= 12:
                    break
        if not non_comment:
            return ","  # default

        sample = "".join(non_comment)
        dialect = csv.Sniffer().sniff(sample)
        delim = dialect.delimiter
        if delim == "#":
            raise ValueError("Sniffer picked '#' (comment char) as delimiter.")
        return delim
    except Exception:
        # Fallback: choose delimiter that yields most columns on first data line
        line = next((l for l in non_comment if l.strip()), None)
        if not line:
            return ","
        return max(candidates, key=lambda d: len(line.split(d)))


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Load NASA/MAST-style CSV with comment headers safely."""
    delimiter = _detect_delimiter_skip_comments(path)
    print(f"[{path.name}] Using delimiter: {repr(delimiter)}")

    # Try fast engine
    try:
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            engine="c",
            comment="#",
            encoding="utf-8-sig",
        )
    except Exception as e:
        print(f"[{path.name}] C engine failed: {e}; retrying with python engine.")
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )

    # Last-resort retry if weirdly few columns
    if df.shape[1] <= 2:
        print(f"[{path.name}] Low column count ({df.shape[1]}). Forcing python engine sep=None.")
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
    # Drop unnamed junk columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


# ----------------- Column mapping -----------------
def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with canonical column names if found."""
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in ALIASES.items():
        for name in aliases:
            if name in df.columns:
                out[canonical] = df[name]
                break
    keep = ["label"] + FEATURE_ORDER + [c for c in REF_KEEP if c != "source"]
    return out[[c for c in keep if c in out.columns]].copy()


def coerce_and_impute(df_mapped: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], dict, pd.Index]:
    """Coerce numerics, cast flags, drop sparse, impute. Returns X, y, feature_cols, meta, row_index."""
    feature_cols = [c for c in FEATURE_ORDER if c in df_mapped.columns]
    if not feature_cols:
        raise ValueError("No feature columns mapped from this dataset. Extend ALIASES to match your file headers.")

    # Coerce numerics
    for c in feature_cols:
        df_mapped[c] = pd.to_numeric(df_mapped[c], errors="coerce")

    # Flags to 0/1
    for flag in ["fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"]:
        if flag in df_mapped.columns:
            df_mapped[flag] = pd.to_numeric(df_mapped[flag], errors="coerce").fillna(0).astype(int)

    # Drop rows with all-NaN features
    df_work = df_mapped.dropna(axis=0, how="all", subset=feature_cols).copy()

    # Drop features >40% missing
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
        # Normalize a few common label spellings to keep consistent across sources
        y_norm = (
            y_ser.str.upper()
            .str.replace("CONFIRM", "CONFIRMED", regex=False)
            .str.replace("FALSEPOSITIVE", "FALSE POSITIVE", regex=False)
            .str.replace("FALSE-POSITIVE", "FALSE POSITIVE", regex=False)
        )
        classes = sorted(y_norm.dropna().unique().tolist())
        if len(classes) >= 2:
            label_map = {cls: i for i, cls in enumerate(classes)}
            y = y_norm.map(label_map).values

    meta = {
        "feature_cols": feature_cols,
        "dropped_sparse_cols": dropped_sparse,
        "label_map": label_map,
    }
    return X, y, feature_cols, meta, df_work.index


def newest_or_none(pattern: str) -> Optional[Path]:
    files = sorted(DATA_DIR.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def process_source(name: str, pattern: str) -> Optional[pd.DataFrame]:
    path = newest_or_none(pattern)
    if not path:
        print(f"[{name}] No files matching {pattern} in {DATA_DIR}")
        return None
    print(f"[{name}] Using file: {path.name}")
    raw = read_csv_robust(path)
    mapped = map_columns(raw)
    if mapped.empty:
        print(f"[{name}] Warning: mapped columns empty. Check headers or extend ALIASES.")
        return None
    mapped["source"] = name
    return mapped


def export_combined(X, y, feature_cols, sources_used, label_map):
    # split (no training)
    if y is not None and len(np.unique(y[~pd.isna(y)])) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train = y_test = None

    # save
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "X_test.npy",  X_test)
    pd.DataFrame(X_train, columns=feature_cols).to_csv(OUT_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test,  columns=feature_cols).to_csv(OUT_DIR / "X_test.csv",  index=False)
    if y_train is not None:
        np.save(OUT_DIR / "y_train.npy", y_train)
        np.save(OUT_DIR / "y_test.npy",  y_test)
        pd.Series(y_train, name="label").to_csv(OUT_DIR / "y_train.csv", index=False)
        pd.Series(y_test,  name="label").to_csv(OUT_DIR / "y_test.csv",  index=False)

    meta = {
        "feature_cols": feature_cols,
        "label_map": label_map,
        "sources_used": sources_used,
        "split": {"test_size": 0.2, "random_state": 42},
    }
    with open(OUT_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved ML-ready combined artifacts to: {OUT_DIR}")
    if label_map:
        print(f"Label classes: {list(label_map.keys())}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")


def main():
    frames = []
    sources_used = []

    for name, pattern in PATTERNS.items():
        df = process_source(name, pattern)
        if df is not None and not df.empty:
            frames.append(df)
            sources_used.append(name)

    if not frames:
        raise SystemExit("No datasets were processed. Ensure your Data/ contains the CSVs.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nCombined shape before coercion: {combined.shape}")
    print(f"Columns present: {combined.columns.tolist()}")

    # Build features + label across combined
    X, y, feature_cols, meta, kept_index = coerce_and_impute(combined)

    # Export
    export_combined(X, y, feature_cols, sources_used, meta["label_map"])


if __name__ == "__main__":
    main()
