# ==========================================
# Space Invaders — LightCurve Model PRO (Final Stable Version)
# ==========================================
import os, warnings, argparse
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")


def extract_bls_features(tic_id, mission="tess", min_period=0.5, max_period=15):
    """Extract transit-like features from a light curve."""
    try:
        lc = lk.search_lightcurve(f"TIC {tic_id}", mission=mission).download()
        if lc is None:
            raise ValueError("no LC found")
        lc = lc.remove_nans().normalize().flatten(window_length=401)
        time, flux = np.array(lc.time.value), np.array(lc.flux.value)
        if len(time) < 100:
            raise ValueError("too short")

        bls = BoxLeastSquares(time, flux)
        periods = np.linspace(min_period, max_period, 800)
        res = bls.power(periods, 0.05)
        best = np.argmax(res.power)
        period = res.period[best]
        duration = res.duration[best] * 24  # convert days → hours
        depth = res.depth[best] * 1e6       # convert fraction → ppm
        snr = max(res.depth_snr[best], 1e-3)
        print(f"✅ TIC{tic_id}: P={period:.4f}d, dur={duration:.2f}h, depth={depth:.0f}ppm, SNR={snr:.2f}")
        return dict(tic_id=int(tic_id), period=period, duration=duration, depth=depth, snr=snr)
    except Exception as e:
        print(f"⚠️ Failed TIC{tic_id}: {e}")
        return None


def train_xgb(X, y, out_dir):
    """Train XGBoost model with small-sample-safe logic."""
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        use_label_encoder=False,
        tree_method="hist"
    )

    if len(X) < 5:
        print("⚠️ Too few samples — training without validation.")
        model.fit(X, y)
        return model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n📊 Evaluation Metrics:")
    print(classification_report(y_test, preds))
    print("Accuracy:", round(accuracy_score(y_test, preds), 3))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tic_ids", nargs="+", required=True)
    parser.add_argument("--mission", default="tess")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min_period", type=float, default=0.5)
    parser.add_argument("--max_period", type=float, default=15)
    args = parser.parse_args()

    labels_path = r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Data\labels2.csv"
    os.makedirs(args.outdir, exist_ok=True)

    # --- Extract features ---
    feats = []
    for tic in args.tic_ids:
        f = extract_bls_features(tic, args.mission, args.min_period, args.max_period)
        if f:
            feats.append(f)

    if not feats:
        raise RuntimeError("No features extracted.")
    df = pd.DataFrame(feats)
    df_path = os.path.join(args.outdir, "lc_features.csv")
    df.to_csv(df_path, index=False)
    print(f"\n📝 Saved features -> {df_path}")

    # --- Load labels ---
    labels = pd.read_csv(labels_path)
    labels.columns = [c.lower().strip() for c in labels.columns]
    if "tic_id" not in labels.columns:
        for c in labels.columns:
            if "tic" in c or "id" == c:
                labels = labels.rename(columns={c: "tic_id"})
                break
    if "label" not in labels.columns:
        raise ValueError("❌ 'label' column not found in labels2.csv")

    merged = pd.merge(df, labels, on="tic_id", how="inner")
    if merged.empty:
        raise RuntimeError("No TICs in labels2.csv matched extracted features.")

    # --- Re-map labels to 0-based integers ---
    y_raw = merged["label"].astype(int)
    unique_vals = sorted(y_raw.unique())
    mapping = {val: i for i, val in enumerate(unique_vals)}
    y = y_raw.map(mapping)
    print(f"ℹ️ Label mapping: {mapping}")

    X = merged[["period", "duration", "depth", "snr"]]
    model = train_xgb(X, y, args.outdir)

    model_path = os.path.join(args.outdir, "lightcurve_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n💾 Saved model -> {model_path}")

    preds = model.predict(X)
    merged["prediction"] = preds
    merged.to_csv(os.path.join(args.outdir, "lc_results.csv"), index=False)
    print(f"✅ Saved results -> {os.path.join(args.outdir, 'lc_results.csv')}")


if __name__ == "__main__":
    main()
