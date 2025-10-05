# --- SpaceInvaders Light-Curve Trainer (stable edition) ---
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")

# ---------- Helpers ----------
def normalize_label(lbl:str)->str:
    s=str(lbl).upper().strip().replace("_"," ")
    if "CONFIRM" in s: return "CONFIRMED"
    if "FALSE" in s or "REFUTED" in s or "FP" in s: return "FALSE POSITIVE"
    return "CANDIDATE"

LABELS={"CONFIRMED":2,"CANDIDATE":1,"FALSE POSITIVE":0}

# ---------- Minimal BLS feature extractor ----------
def extract_features(time,flux):
    import numpy as np
    from astropy.timeseries import BoxLeastSquares
    m=np.isfinite(time)&np.isfinite(flux)
    t,f=time[m],flux[m]/np.nanmedian(flux[m])
    bls=BoxLeastSquares(t,f)
    res=bls.autopower(0.1)
    i=int(np.nanargmax(res.power))
    return float(res.period[i]),float(res.duration[i]*24),float(res.depth[i]*1e6),float(np.nanmax(res.power))

# ---------- Model ----------
def train_xgb(X,y,seed=42):
    import xgboost as xgb
    # force labels to 0..n-1
    classes,inv=np.unique(y,return_inverse=True)
    y=inv
    dtrain=xgb.DMatrix(X,label=y)
    params={"objective":"multi:softprob","num_class":len(classes),"eta":0.05,
            "max_depth":5,"subsample":0.9,"colsample_bytree":0.9,"seed":seed}
    booster=xgb.train(params,dtrain,num_boost_round=400,verbose_eval=False)
    return booster,classes

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels_csv",required=True)
    ap.add_argument("--outdir",default="Processed/LC_Model")
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    # Load example light curves
    import lightkurve as lk
    tics=["150428135","307210830","123456789"]
    rows=[]
    for tid in tics:
        lc=lk.search_lightcurve(f"TIC {tid}",mission="TESS").download().remove_nans()
        p,dur,depth,snr=extract_features(lc.time.value,lc.flux.value)
        rows.append({"source":f"TIC{tid}","period":p,"dur_h":dur,"depth_ppm":depth,"snr":snr})
        print(f"✅ {tid}: {p:.2f}d {dur:.2f}h {depth:.0f}ppm snr={snr:.2f}")

    feats=pd.DataFrame(rows)
    labels=pd.read_csv(args.labels_csv)
    labels["label"]=labels["label"].apply(normalize_label)
    df=feats.merge(labels,on="source",how="left").dropna()
    X=df[["period","dur_h","depth_ppm","snr"]].to_numpy()
    y=df["label"].map(LABELS).to_numpy()

    if len(np.unique(y))<2:
        print("⚠️ Not enough classes — training anyway.")
    model,classes=train_xgb(X,y)
    joblib.dump({"model":model,"classes":classes},out/"model.pkl")
    feats.to_csv(out/"lc_features.csv",index=False)
    print("💾 Model + features saved!")

if __name__=="__main__": main()
