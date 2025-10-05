# =========================================================
# Space Invaders — Post-process KOI predictions
# - Validates vs koi_disposition (CONFIRMED/CANDIDATE/FP)
# - Creates a high-confidence shortlist
# - Augments with RA/Dec and exports JSON for the UI
# =========================================================
# Run (PowerShell):
#   python Code\postprocess_koi_predictions.py `
#     --pred "C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Predictions\koi_500_predictions.csv" `
#     --outdir "C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Predictions"
#
# Outputs:
#   - validation_report.txt
#   - koi_shortlist.csv  (top candidates based on Prob_class_1 >= threshold)
#   - koi_augmented.csv  (preds + RA/Dec + handy fields)
#   - planets_for_ui.json (ready for React map)
# =========================================================

import os, io, argparse, json, requests
import pandas as pd
import numpy as np

CLASSIC_API = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

# Columns we will try to fetch to augment the CSV (RA/Dec names vary by table)
AUGMENT_COLS = [
    "kepid", "koi_prad", "koi_period", "koi_teq", "koi_insol",
    "ra", "dec", "koi_srad", "koi_steff", "koi_slogg"
]

def fetch_augments_for_kepids(kepids):
    """Fetch RA/Dec and a few handy cols for a list of kepid values via classic API."""
    if len(kepids) == 0:
        return pd.DataFrame(columns=AUGMENT_COLS)

    # Build a comma-separated kepid list (limited per request)
    # We’ll chunk to be safe
    all_rows = []
    chunk_size = 200
    for i in range(0, len(kepids), chunk_size):
        chunk = kepids[i:i+chunk_size]
        where = " or ".join([f"kepid={int(k)}" for k in chunk if pd.notna(k)])
        params = {
            "table": "koi",
            "format": "csv",
            "select": ",".join(AUGMENT_COLS),
            "where": where
        }
        r = requests.get(CLASSIC_API, params=params, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame(columns=AUGMENT_COLS)
    out = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["kepid"])
    return out

def label_to_index(x):
    x = str(x).strip().upper()
    if x.startswith("FALSE"):
        return -1
    if x.startswith("CONFIRM"):
        return 1
    if x.startswith("CANDID"):
        return 0
    return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Path to koi_500_predictions.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for reports/JSON")
    ap.add_argument("--cand_thresh", type=float, default=0.80, help="Prob threshold for CANDIDATE shortlist (Prob_class_1)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load predictions ---
    df = pd.read_csv(args.pred)
    # Try to find probability column for class 1 (candidate)
    prob_cand_col = None
    for c in df.columns:
        if c.lower() in ["prob_class_1", "prob_candidate", "prob_cand", "prob_cand_1"]:
            prob_cand_col = c
            break

    # --- Validation vs koi_disposition (if available) ---
    # Map koi_disposition to [-1,0,1]
    df["koi_disp_idx"] = df["koi_disposition"].map(label_to_index)
    has_truth = df["koi_disp_idx"].notna().any()

    # Best-effort mapping for predictions (already has Pred_Class_Index/Pred_Label)
    def pred_index_from_row(r):
        # Prefer Pred_Class_Index if it’s numeric
        if "Pred_Class_Index" in r and pd.notna(r["Pred_Class_Index"]):
            try:
                return int(r["Pred_Class_Index"])
            except Exception:
                pass
        # Else map by label
        lbl = str(r.get("Pred_Label", "")).upper()
        if lbl.startswith("FALSE"): return -1
        if lbl.startswith("CONFIRM"): return 1
        if lbl.startswith("CANDID"): return 0
        return np.nan

    df["pred_idx"] = df.apply(pred_index_from_row, axis=1)

    # Simple validation report
    report_path = os.path.join(args.outdir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Space Invaders — KOI bulk scoring validation\n")
        f.write(f"Source predictions: {args.pred}\n")
        f.write(f"Rows: {len(df)}\n\n")

        # Distribution of predictions
        f.write("Prediction distribution (by Pred_Label):\n")
        f.write(df["Pred_Label"].value_counts(dropna=False).to_string() + "\n\n")

        # If truth present, compute accuracy-like metrics (coarse)
        if has_truth:
            valid = df.dropna(subset=["koi_disp_idx", "pred_idx"])
            if len(valid) > 0:
                acc = (valid["koi_disp_idx"] == valid["pred_idx"]).mean()
                f.write(f"Rows with truth labels: {len(valid)}\n")
                f.write(f"Label agreement rate (Pred vs koi_disposition): {acc:.3f}\n")
                f.write("\nConfusion (rows per pair):\n")
                conf = (valid.groupby(["koi_disp_idx","pred_idx"])
                             .size()
                             .rename("count")
                             .reset_index())
                f.write(conf.to_string(index=False) + "\n\n")
            else:
                f.write("Insufficient overlap to compute agreement.\n\n")
        else:
            f.write("No koi_disposition labels available for validation in this file.\n\n")

        # Shortlist summary
        if prob_cand_col and prob_cand_col in df.columns:
            shortlist = df[df[prob_cand_col] >= args.cand_thresh].copy()
            f.write(f"Shortlist (Prob candidate >= {args.cand_thresh}): {len(shortlist)} rows\n")
        else:
            f.write("Probabilities not present; shortlist by probability skipped.\n")

    print(f"🧾 Wrote validation report → {report_path}")

    # --- Shortlist high-confidence candidates ---
    shortlist_path = os.path.join(args.outdir, "koi_shortlist.csv")
    if prob_cand_col and prob_cand_col in df.columns:
        shortlist = df[df[prob_cand_col] >= args.cand_thresh].copy()
        shortlist = shortlist.sort_values(prob_cand_col, ascending=False)
        shortlist.to_csv(shortlist_path, index=False)
        print(f"⭐ Saved shortlist → {shortlist_path}")
    else:
        print("⚠️ No probability column found; skipping shortlist. (Add Prob_class_1 during training to enable this.)")

    # --- Augment with RA/Dec for UI ---
    # Fetch RA/Dec and selected physical columns for the kepid list we have
    kepids = [int(k) for k in df["kepid"].dropna().unique().tolist()]
    aug = fetch_augments_for_kepids(kepids)

    # Merge and save augmented CSV
    merged = df.merge(aug, on="kepid", how="left", suffixes=("", "_aug"))
    augmented_csv_path = os.path.join(args.outdir, "koi_augmented.csv")
    merged.to_csv(augmented_csv_path, index=False)
    print(f"🛰  Saved augmented CSV → {augmented_csv_path}")

    # --- Export compact JSON for the React map ---
    # Minimal fields; rename for clarity; ensure floats for serializing
    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    records = []
    for _, r in merged.iterrows():
        records.append({
            "kepid": int(r["kepid"]) if pd.notna(r["kepid"]) else None,
            "name": r["kepler_name"] if pd.notna(r["kepler_name"]) and str(r["kepler_name"]).strip() != "" else r.get("kepoi_name", None),
            "koi_disposition": r.get("koi_disposition", None),
            "pred_label": r.get("Pred_Label", None),
            "pred_index": int(r["Pred_Class_Index"]) if pd.notna(r.get("Pred_Class_Index", np.nan)) else None,
            "prob_class_0": safe_float(r.get("Prob_class_0", None)),
            "prob_class_1": safe_float(r.get("Prob_class_1", None)),
            "prob_class_2": safe_float(r.get("Prob_class_2", None)),
            "period_days": safe_float(r.get("koi_period", None)),
            "planet_radius_re": safe_float(r.get("koi_prad", None)),
            "teq_k": safe_float(r.get("koi_teq", None)),
            "insol_earth": safe_float(r.get("koi_insol", None)),
            "star_teff_k": safe_float(r.get("koi_steff", None)),
            "star_logg": safe_float(r.get("koi_slogg", None)),
            "star_radius_rsun": safe_float(r.get("koi_srad", None)),
            "ra": safe_float(r.get("ra", None)),
            "dec": safe_float(r.get("dec", None)),
        })

    ui_json_path = os.path.join(args.outdir, "planets_for_ui.json")
    with open(ui_json_path, "w", encoding="utf-8") as f:
        json.dump({"items": records}, f, indent=2)
    print(f"🗺  Saved React-ready JSON → {ui_json_path}")

if __name__ == "__main__":
    main()
