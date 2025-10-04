import pandas as pd
import re
from pathlib import Path
import json

BASE = Path(r"C:\Users\write\OneDrive\Documentos\GitHub\SpaceInvaders-NASASpaceApps")
PRED_IN  = BASE / r"Processed\Combined\predictions.csv"
META_IN  = BASE / r"Processed\Combined\feature_metadata.json"
PRED_OUT = BASE / r"Processed\Combined\predictions_tidy.csv"

# Load
df = pd.read_csv(PRED_IN)
meta = json.loads(META_IN.read_text(encoding="utf-8"))
feature_cols = meta.get("feature_cols", [])

# Detect probability columns
prob_cols = [c for c in df.columns if c.startswith("Prob_")]

# Normalize class names to a small set
# Map many catalog spellings into: CONFIRMED, CANDIDATE, FALSE POSITIVE
ALIASES = {
    r"CONFIRM(ED|EDED|)": "CONFIRMED",
    r"(FALSE[_\s-]?POSITIVE|FP|REFUTED)": "FALSE POSITIVE",
    r"(CANDIDATE|PC|KP|CP|APC)": "CANDIDATE",  # many catalogs use PC/APC for "planet candidate"
}

def normalize_name(name: str) -> str:
    # Strip "Prob_" prefix
    cls = name[5:] if name.startswith("Prob_") else name
    cls_u = cls.upper().strip()
    for pat, target in ALIASES.items():
        if re.fullmatch(pat, cls_u):
            return target
    return cls_u

# Collapse probabilities by normalized class
collapsed = {}
for c in prob_cols:
    norm = "Prob_" + normalize_name(c)
    if norm not in collapsed:
        collapsed[norm] = df[c].fillna(0.0)
    else:
        collapsed[norm] = collapsed[norm] + df[c].fillna(0.0)

# Build tidy frame
tidy = pd.DataFrame(collapsed)
# If none existed (rare), create zero columns for standard classes
for std in ["Prob_CONFIRMED","Prob_CANDIDATE","Prob_FALSE_POSITIVE"]:
    if std not in tidy.columns:
        tidy[std] = 0.0

# Re-normalize rows to sum to 1 (in case of merged duplicates)
row_sum = tidy.sum(axis=1).replace(0, 1.0)
tidy = tidy.div(row_sum, axis=0)

# Pred label and confidence from tidy
tidy_labels = []
tidy_conf   = []
for _, r in tidy.iterrows():
    best = r.idxmax()              # e.g., "Prob_CONFIRMED"
    tidy_labels.append(best.replace("Prob_",""))
    tidy_conf.append(float(r.max()))

tidy["pred_label"] = tidy_labels
tidy["pred_confidence"] = tidy_conf

# Add top3 text summary
def topk_series(s, k=3):
    items = sorted(((col.replace("Prob_",""), float(s[col])) for col in s.index if col.startswith("Prob_")),
                   key=lambda x: x[1], reverse=True)[:k]
    return " | ".join([f"{n}: {p:.3f}" for n,p in items])

tidy["top3"] = tidy.apply(lambda r: topk_series(r), axis=1)

# Keep a few context features for interpretability
context_cols = [c for c in feature_cols if c in df.columns]
context_keep = ["koi_period","koi_prad","koi_model_snr","koi_steff","koi_srad","koi_insol","koi_teq"]
context_keep = [c for c in context_keep if c in df.columns]

final_cols = (
    ["pred_label","pred_confidence","top3"] +
    ["Prob_CONFIRMED","Prob_CANDIDATE","Prob_FALSE_POSITIVE"] +
    context_keep
)

out = pd.concat([tidy[["pred_label","pred_confidence","top3",
                       "Prob_CONFIRMED","Prob_CANDIDATE","Prob_FALSE_POSITIVE"]],
                 df[context_keep]], axis=1)

out.to_csv(PRED_OUT, index=False)
print(f"✅ Wrote tidy predictions -> {PRED_OUT}")
print(out.head(10))
