# ============================================
# Space Invaders — Train Model from Processed Artifacts (robust XGB)
# ============================================

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

DATA_DIR = Path(r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps\Processed\Combined")

# ------------------------ Load artifacts ------------------------
def load_artifacts():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")

    y_train = y_test = None
    if (DATA_DIR / "y_train.npy").exists():
        y_train = np.load(DATA_DIR / "y_train.npy", allow_pickle=True)
        y_test  = np.load(DATA_DIR / "y_test.npy", allow_pickle=True)

    meta_path = DATA_DIR / "feature_metadata.json"
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    label_map = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_names = meta.get("feature_cols", feature_names)
        label_map = meta.get("label_map", None)

    return X_train, X_test, y_train, y_test, feature_names, label_map

# ------------------------ Helpers ------------------------
def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def save_feature_importance_generic(model, feature_names, out_csv: Path):
    fi = None
    # sklearn tree models
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    # logistic regression
    elif hasattr(model, "coef_"):
        coef = model.coef_
        fi = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    # xgboost Booster wrapper
    elif hasattr(model, "get_fscore_dict"):
        fscore = model.get_fscore_dict()
        # map to feature_names by index; xgb uses "f0", "f1", ...
        imp = np.zeros(len(feature_names), dtype=float)
        for k, v in fscore.items():
            if k.startswith("f"):
                try:
                    idx = int(k[1:])
                    if 0 <= idx < len(imp):
                        imp[idx] = float(v)
                except:
                    pass
        fi = imp
    if fi is None:
        return False
    df = pd.DataFrame({"feature": feature_names, "importance": fi})
    df.sort_values("importance", ascending=False, inplace=True)
    df.to_csv(out_csv, index=False)
    return True

# ------------------------ Models ------------------------
def build_model(name: str, seed: int, class_weight=None):
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight=class_weight
        )
    if name == "logreg":
        return LogisticRegression(
            multi_class="auto",
            class_weight=class_weight,
            max_iter=1000,
            n_jobs=-1
        )
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1
        )
    if name == "xgb":
        # We'll handle XGB specially in train_and_eval() for maximum compatibility.
        return "XGB_PLACEHOLDER"
    raise ValueError("Unknown model. Use rf | logreg | lgbm | xgb.")

# ------------------------ XGB Fallback Wrapper ------------------------
class XGBBoosterWrapper:
    """Light wrapper to unify prediction API when we train with xgboost.train()."""
    def __init__(self, booster, num_class: int):
        self.booster = booster
        self.num_class = num_class

    def predict(self, X):
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        if self.num_class <= 2:
            # binary: output margin -> probability via logistic
            prob = self.booster.predict(dm)
            # xgb.train binary may return prob of positive class
            pred = (prob >= 0.5).astype(int)
            return pred
        else:
            prob = self.booster.predict(dm)  # shape (n, num_class)
            pred = np.argmax(prob, axis=1)
            return pred

    def get_fscore_dict(self):
        # returns dict like {"f0": score, ...}
        try:
            return self.booster.get_fscore()
        except Exception:
            # newer API: get_score
            try:
                return self.booster.get_score(importance_type="weight")
            except Exception:
                return {}

# ------------------------ Training ------------------------
def train_and_eval(model_name, X_train, y_train, X_test, y_test, seed, out_dir: Path):
    if model_name == "xgb":
        # Try sklearn API first; on failure, fallback to xgboost.train
        import xgboost as xgb
        cw = compute_class_weights(y_train) if y_train is not None else None
        sample_weight = np.vectorize(lambda c: cw[int(c)])(y_train) if cw else None
        eval_set = [(X_test, y_test)] if y_test is not None else None

        try:
            # --- sklearn wrapper path (newer xgboost) ---
            model = xgb.XGBClassifier(
                n_estimators=2000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                eval_metric="mlogloss"
            )
            # Try the "new" signature
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=100 if y_test is not None else None
            )
            used_fallback = False

        except TypeError:
            # --- hard fallback: native xgboost.train (works on very old versions) ---
            warnings.warn("XGB sklearn API lacks early_stopping/callbacks. Falling back to xgboost.train().", RuntimeWarning)
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
            dvalid = xgb.DMatrix(X_test, label=y_test) if y_test is not None else None

            # identify number of classes
            classes = np.unique(y_train)
            num_class = int(classes.max() - classes.min() + 1) if len(classes) > 2 else 2
            # heuristic: if labels aren't {0..k-1}, map them
            label_values = np.unique(y_train)
            if not np.array_equal(label_values, np.arange(label_values.min(), label_values.min()+len(label_values))):
                # remap to 0..k-1
                mapping = {v:i for i,v in enumerate(label_values)}
                y_train_m = np.vectorize(lambda z: mapping[z])(y_train)
                dtrain = xgb.DMatrix(X_train, label=y_train_m, weight=sample_weight)
                if y_test is not None:
                    y_test_m = np.vectorize(lambda z: mapping[z])(y_test)
                    dvalid = xgb.DMatrix(X_test, label=y_test_m)
                classes = np.unique(y_train_m)
                num_class = len(classes)

            params = {
                "seed": seed,
                "eta": 0.03,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "eval_metric": "mlogloss",
                "objective": "binary:logistic" if num_class <= 2 else "multi:softprob",
            }
            if num_class > 2:
                params["num_class"] = num_class

            evals = [(dtrain, "train")]
            if dvalid is not None:
                evals.append((dvalid, "valid"))

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=2000,
                evals=evals,
                early_stopping_rounds=100 if dvalid is not None else 0,
                verbose_eval=False
            )
            model = XGBBoosterWrapper(booster, num_class=num_class)
            used_fallback = True

        # Evaluate
        metrics = {}
        if y_test is not None:
            y_pred = model.predict(X_test)
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
            metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
            metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
        else:
            metrics["note"] = "No labels found; trained without evaluation split."
            y_pred = None

        return model, metrics, y_pred

    # ----- non-XGB paths (RF / LOGREG / LGBM) -----
    cw = compute_class_weights(y_train) if y_train is not None else None
    model = build_model(model_name, seed=42, class_weight=cw if model_name in ["rf", "logreg"] else None)

    if model_name == "lgbm":
        import lightgbm as lgb
        fit_params = {}
        if y_test is not None:
            fit_params.update({
                "eval_set": [(X_test, y_test)],
                "eval_metric": "multi_logloss",
                "callbacks": [lgb.early_stopping(stopping_rounds=100, verbose=False)]
            })
        if cw:
            model.set_params(class_weight=cw)
        model.fit(X_train, y_train, **fit_params)
    else:
        model.fit(X_train, y_train)

    metrics = {}
    if y_test is not None:
        y_pred = model.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
    else:
        metrics["note"] = "No labels found; trained without evaluation split."
        y_pred = None

    return model, metrics, y_pred

# ------------------------ Main ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", help="rf | logreg | lgbm | xgb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, feature_names, label_map = load_artifacts()
    print(f"✅ Loaded data from: {DATA_DIR}")

    out_dir = DATA_DIR / "Models" / args.model.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, metrics, _ = train_and_eval(args.model, X_train, y_train, X_test, y_test, args.seed, out_dir)

    # Save model (Booster wrapper is pickleable)
    joblib.dump(model, out_dir / "model.pkl")
    print(f"✅ Model saved -> {out_dir / 'model.pkl'}")

    # Feature importance
    if save_feature_importance_generic(model, feature_names, out_dir / "feature_importance.csv"):
        print(f"📝 Feature importance -> {out_dir / 'feature_importance.csv'}")
    else:
        print("ℹ️ Feature importance not available for this model.")

    # Metrics
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"📊 Metrics -> {out_dir / 'metrics.json'}")

    if y_test is not None and "accuracy" in metrics:
        print("\n=== Evaluation Summary ===")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"F1 (macro):         {metrics['f1_macro']:.4f}")
    else:
        print("⚠️ No labels found, skipping summary.")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
