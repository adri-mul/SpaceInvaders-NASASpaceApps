🌌 Space Invaders — AI-Powered Exoplanet Discovery
🚀 NASA Space Apps Challenge 2025

Team: Space Invaders
Theme: Exoplanet Exploration & Detection using AI/ML

🧠 Overview

Space Invaders is an open-source pipeline that unifies and analyzes NASA’s exoplanet datasets (Kepler KOI, TESS TOI, and K2 Planet Candidates).
We designed a clean, ML-ready dataset builder and a visualization platform to accelerate the discovery and classification of exoplanets using modern AI.

🛰️ Problem Statement

NASA’s exoplanet catalogs contain thousands of entries collected from multiple missions (Kepler, K2, TESS).
However:

Each mission stores data in different formats, columns, and delimiters.

Manual preprocessing is tedious and inconsistent.

Many datasets contain metadata noise and commented headers that break automated parsing.

Our project unifies these datasets and prepares them for machine learning classification, allowing scientists and citizen developers to build models that identify new exoplanets faster.

🔬 Our Solution

We created a universal Python data-preparation engine that:

Automatically detects and cleans NASA CSVs (ignores # metadata, guesses delimiters, fixes malformed rows).

Normalizes columns across missions to a shared astrophysical feature schema.

Imputes missing data and converts categorical values (e.g., CONFIRMED / CANDIDATE / FALSE POSITIVE) to numeric labels.

Combines Kepler + TESS + K2 into a single unified dataset for AI/ML pipelines.

Exports clean artifacts (.npy, .csv, and feature_metadata.json) ready for models such as XGBoost, LightGBM, or PyTorch Tabular NNs.

📂 Project Structure
SpaceInvaders-NASASpaceApps/
│
├── Code/
│   ├── Prepare_All_Combined.py     ← Unified data-prep script (main entry)
│   ├── CumulativeDataAnlysis/…     ← Kepler (KOI) processing
│   ├── K2DataAnlysis/…             ← K2 data prep utilities
│   └── TOIDataAnlysis/…            ← TESS (TOI) data prep utilities
│
├── Data/
│   ├── cumulative_2025.10.04_09.06.58.csv
│   ├── TOI_2025.10.04_09.07.03.csv
│   └── k2pandc_2025.10.04_09.07.07.csv
│
└── Processed/
    ├── Combined/
    │   ├── X_train.npy
    │   ├── y_train.npy
    │   ├── X_test.npy
    │   ├── feature_metadata.json
    │   └── …
    └── (per-mission subfolders)

🧩 Key Features

Universal Loader – Handles any NASA-style CSV (auto-detects delimiter, ignores metadata lines).

Cross-Mission Normalization – Maps all missions to 15 core features (orbital, stellar, flag parameters).

Noise Filtering & Imputation – Drops > 40 %-missing columns and fills others with median values.

Consistent Label Encoding – CONFIRMED = 1, CANDIDATE = 0, FALSE POSITIVE = −1 (by default).

Automatic Split & Export – Saves train/test sets for immediate model training.

🧠 Recommended ML Models
Model	Library	Why Use It
XGBoost	xgboost	Best accuracy for tabular NASA data
LightGBM	lightgbm	Fast & memory-efficient
Random Forest	sklearn	Interpretable baseline
Neural Network (MLP)	tensorflow / torch	For non-linear astrophysical patterns
⚙️ Setup & Usage

Clone Repo

git clone https://github.com/yourusername/SpaceInvaders-NASASpaceApps.git
cd SpaceInvaders-NASASpaceApps


Install Dependencies

pip install pandas numpy scikit-learn xgboost lightgbm


Run Combined Data Prep

python Code/Prepare_All_Combined.py


Output Example

✅ Saved ML-ready combined artifacts to: Processed/Combined
Label classes: ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
Features used (15): [...]

📊 Next Steps

Integrate real-time training using XGBoost.

Build an interactive 3D space map (React + Three.js) to visualize exoplanet locations.

Add NASA Exoplanet Archive API fetcher for live data updates.

🧑‍💻 Team Roles
Member	Role	Focus
Shamliki Sharma	Lead Developer	Data pipeline, ML integration
[Teammate 2]	Data Scientist	Feature engineering & analysis
[Teammate 3]	Frontend Engineer	3D map & visualization UI
[Teammate 4]	Research Lead	Dataset curation & validation
🛰️ Datasets Used

Kepler Exoplanet Cumulative Table — NASA Exoplanet Archive

TESS Objects of Interest (TOI) — NASA MAST

K2 Planet Candidates and Confirmed Planets — NASA Exoplanet Archive

📜 License

MIT License — Free for research and educational use.

✨ Acknowledgments

Special thanks to NASA, Caltech IPAC, and the Space Apps organizers for open-data access and community support.
AND OFCOURSE CHATGPT!!!!

“Somewhere, something incredible is waiting to be known.” — Carl Sagan


Generative AI was used to make this README
