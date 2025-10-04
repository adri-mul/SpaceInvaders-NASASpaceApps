# Code/build_stars_catalog.py
import csv
import json
import os
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]   # repo root
DATA = BASE / "Data"
PUBLIC = BASE / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)

FILES = {
    "KOI": "cumulative_*.csv",          # Kepler cumulative table
    "TOI": "TOI_*.csv",                 # TESS TOI
    "K2" : "k2pandc_*.csv",             # K2 P&C
}

def newest(globpat):
    c = sorted(DATA.glob(globpat), key=os.path.getmtime, reverse=True)
    return c[0] if c else None

def read_csv_robust(path: Path) -> pd.DataFrame:
    # ignore lines starting with '#'
    try:
        # detect delimiter from non-comment lines
        non_comment = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for _ in range(2000):
                line = f.readline()
                if not line:
                    break
                if not line.lstrip().startswith("#") and line.strip():
                    non_comment.append(line)
                if len(non_comment) >= 10:
                    break
        delim = ","
        if non_comment:
            try:
                dialect = csv.Sniffer().sniff("".join(non_comment))
                if dialect.delimiter != "#":
                    delim = dialect.delimiter
            except Exception:
                pass
        df = pd.read_csv(path, delimiter=delim, comment="#", encoding="utf-8-sig", engine="c")
    except Exception:
        df = pd.read_csv(path, sep=None, comment="#", encoding="utf-8-sig", engine="python", on_bad_lines="skip")

    # drop junk cols
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df

def extract_names(df: pd.DataFrame, source: str):
    """
    Return a list of dicts with star identifiers / names.
    We try common columns across the three catalogs.
    """
    cols = df.columns.str.lower()

    results = []

    # Prefer explicit host names if present
    # (Some exports include 'hostname' or mission-specific names)
    candidates_name = [c for c in ["hostname", "host_name", "star_name", "k2_name", "kepler_name"] if c in cols]
    candidates_ids  = [c for c in ["kepid", "tic id", "tic_id", "ticid", "epic", "epic_id"] if c in cols]

    def getcol(name):
        # find case-insensitively
        matches = [c for c in df.columns if c.lower() == name]
        return matches[0] if matches else None

    # name first
    for cname in candidates_name:
        col = getcol(cname)
        if col:
            for v in df[col].dropna().astype(str).unique().tolist():
                v = v.strip()
                if v:
                    results.append({"source": source, "id": v, "display": v})
            break  # use the first found name column

    # if no names, fall back to IDs (TIC, EPIC, KEPID)
    if not results:
        for cid in candidates_ids:
            col = getcol(cid)
            if col:
                for v in df[col].dropna().astype(str).unique().tolist():
                    v = v.strip()
                    if v:
                        # Give a nicer display name
                        if "tic" in cid:
                            disp = f"TIC {v}"
                        elif "kepid" in cid:
                            disp = f"Kepler ID {v}"
                        elif "epic" in cid:
                            disp = f"EPIC {v}"
                        else:
                            disp = v
                        results.append({"source": source, "id": v, "display": disp})
                break

    # As an extra fallback, use object names if that’s all we have (e.g., TOI, KOI)
    if not results:
        for cname in ["toi", "toi_name", "kepoi_name", "object_name"]:
            col = getcol(cname)
            if col:
                for v in df[col].dropna().astype(str).unique().tolist():
                    v = v.strip()
                    if v:
                        results.append({"source": source, "id": v, "display": v})
                break

    return results

def main():
    all_items = []

    for src, pat in FILES.items():
        p = newest(pat)
        if not p:
            print(f"[{src}] no files found for pattern {pat}")
            continue
        print(f"[{src}] using {p.name}")
        df = read_csv_robust(p)
        items = extract_names(df, src)
        print(f"[{src}] extracted {len(items)} star entries")
        all_items.extend(items)

    # de-duplicate by display name
    seen = set()
    unique = []
    for it in all_items:
        key = it["display"].upper()
        if key not in seen:
            seen.add(key)
            # Add a sensible search query for SerpAPI (mission tag helps disambiguate)
            qtag = {"KOI": "Kepler star", "TOI": "TESS star", "K2": "K2 star"}.get(it["source"], "exoplanet host")
            unique.append({
                **it,
                "search_query": f"{it['display']} {qtag}"
            })

    out_path = PUBLIC / "stars.json"
    out_path.write_text(json.dumps(unique, indent=2), encoding="utf-8")
    print(f"✅ wrote {len(unique)} unique stars -> {out_path}")

if __name__ == "__main__":
    main()
