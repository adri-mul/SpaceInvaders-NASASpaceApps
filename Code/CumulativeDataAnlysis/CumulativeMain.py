# ==============================
# Space Invaders — Universal CSV Loader
# ==============================

import pandas as pd
import os
import csv

# --- Locate CSV relative to this script ---
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "..", "..", "Data", "cumulative_2025.10.04_09.06.58.csv")

# --- Verify file existence ---
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found at: {csv_path}")

# --- Attempt to sniff delimiter ---
try:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
    delimiter = dialect.delimiter
    print(f"Detected delimiter: {repr(delimiter)}")
except Exception as e:
    print(f"⚠️ Could not auto-detect delimiter ({e}), defaulting to comma.")
    delimiter = ","

# --- Load CSV safely ---
try:
    CumulativeDataCSV = pd.read_csv(
        csv_path,
        delimiter=delimiter,
        engine="python",      # tolerant to uneven rows
        comment="#",          # ignore commented metadata
        encoding="utf-8-sig",
        on_bad_lines="warn"   # change to "skip" if you prefer silent drops
    )
except Exception as e:
    print(f"⚠️ Fallback triggered due to: {e}")
    CumulativeDataCSV = pd.read_csv(
        csv_path,
        sep=None,             # let pandas guess
        engine="c",           # faster C engine fallback
        encoding="utf-8-sig",
        on_bad_lines="skip"
    )

# --- Display summary ---
print(f"✅ Loaded CSV: {os.path.basename(csv_path)}")
print(f"Shape: {CumulativeDataCSV.shape}")
print(CumulativeDataCSV.head(10))
