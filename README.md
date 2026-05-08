# 🏎️ F1 Tyre Degradation & Lap Time Prediction (2024 Season)

> A complete machine learning pipeline for analysing tyre degradation behaviour and predicting lap times across the 2024 Formula 1 season — built as a deep learning final-year project.

**Tools:** Python · FastF1 · pandas · scikit-learn · TensorFlow · PyTorch · matplotlib · seaborn · Google Colab  
**Data:** ~26,000 laps · 24 drivers · 24 Grand Prix events · 2024 F1 season

---

## 📌 Project Overview

This project builds an end-to-end predictive modelling system for F1 tyre degradation, progressing from raw telemetry data through to a hybrid stacking ensemble model — the same modelling philosophy used in real-world race strategy tooling.

The pipeline covers:
- Official lap-level data acquisition via **FastF1**
- Robust **data cleaning and stint construction**
- **Feature engineering** focused on degradation signals
- **Exploratory data analysis** at per-race granularity
- Baseline through advanced **ML models** (Linear, Ridge, Random Forest, Gradient Boosting)
- **Deep learning** architectures (MLP, LSTM seq2seq, GRU many-to-one)
- **Cross-race generalisation** via leave-one-event-out validation
- A final **hybrid stacking ensemble** combining LR + RF + MLP

---

## 📂 Repository Structure

```
f1-tyre-degradation-2024/
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb        # FastF1 extraction, all 24 races
│   ├── 02_data_cleaning.ipynb           # Stint construction, outlier removal
│   ├── 03_feature_engineering.ipynb     # Degradation signals, rolling stats
│   ├── 04_eda.ipynb                     # Per-race visualisations
│   ├── 05_baseline_models.ipynb         # Linear & Ridge Regression
│   ├── 06_advanced_ml.ipynb             # Random Forest & Gradient Boosting
│   ├── 07_deep_learning.ipynb           # MLP, LSTM (seq2seq), GRU
│   └── 08_stacking_ensemble.ipynb       # Final hybrid model
│
├── data/
│   ├── raw/                             # Raw FastF1 CSVs (gitignored)
│   └── clean/
│       ├── clean_laps_2024.csv
│       └── clean_laps_2024_fe.csv       # With engineered features
│
├── figures/                             # All plots and visualisations
├── final_report.pdf                     # Full project report
└── README.md
```



---

## 🔧 Key Pipeline Steps

### 1. Data Acquisition
- Loaded all 24 race sessions (`'R'`) of the 2024 F1 season via FastF1
- Retained: `driver`, `team`, `lap_number`, `LapTime`, `compound`, `IsPitLap`, `Stint`, `LapStartTime`
- Pre-season testing excluded; championship round number preserved for ordering

### 2. Data Cleaning & Stint Construction
- Boolean coercion of `is_pit` to prevent string-based logical errors
- Stints constructed **before** removing pit/out-laps to preserve tyre boundaries
- Removed: pit laps (`p`) and out-laps (`p+1`); retained in-laps (`p-1`) for race-pace signal
- Filtered stints with `stint_length < 3`; outliers removed at `mean + 4σ` per driver

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `stint_mean` | Mean lap time within a driver's stint |
| `lap_time_rel` | Lap time minus stint mean — isolates pace variation |
| `lap_time_norm` | Normalised lap time — enables cross-race comparison |
| `rt_3` / `rt_5` | Rolling averages (3-lap / 5-lap windows) |
| `lap_time_diff` | Lap-to-lap delta — captures immediate degradation signal |
| `degradation_from_first` | Cumulative degradation relative to stint lap 1 |
| `stint_progress` | Normalised position within stint (0–1) |

### 4. Modelling Results

#### Baseline (without enhanced features)
| Model | MAE (s) |
|---|---|
| Linear Regression | 3.71 |
| Ridge Regression | 3.71 |

#### With Enhanced Features
| Model | MAE (s) | R² |
|---|---|---|
| Linear Regression | ~0.88 | ~0.98 |
| Ridge Regression | ~0.88 | ~0.98 |
| Lasso Regression | ~0.88 | ~0.98 |
| **Random Forest** | **~0.54** | **~0.99** |
| HistGradient Boosting | ~0.64 | ~0.99 |

#### Deep Learning
| Model | MAE (s) | Notes |
|---|---|---|
| MLP | ~1.32 | Best for single-lap pointwise prediction |
| LSTM (seq2seq) | ~8–9 | Full stint-curve forecasting |
| GRU (many-to-one) | ~8.18 | Efficient recurrent alternative |

#### Final Stacking Ensemble (LR + RF + MLP → Ridge meta-learner)
| Model | MAE (s) | RMSE (s) | R² |
|---|---|---|---|
| **Stacked Ensemble** | **~0.001** | **~0.002** | **1.000** |

### 5. Cross-Race Generalisation (Leave-One-Event-Out)

Simulates deploying the model on an **unseen Grand Prix** — the realistic use case.

- **Best generalisation:** Abu Dhabi (~3.3 s), Saudi Arabia (~3.5 s), Miami (~4.5 s)
- **Most challenging:** Austria (~20 s), Azerbaijan (~18 s), Belgium (~18 s)
- Circuits with aggressive strategies, extreme temperature sensitivity, or pace variability are inherently harder to predict

---

## 📊 Key Visualisations

- Per-race **lap time distributions** and **degradation curves by compound**
- **Cross-race MAE bar charts** (Ridge, Random Forest, Gradient Boosting)
- **Feature importance tables** from Gradient Boosting
- **Residual histograms** for all models
- **Actual vs Predicted scatter plots** and stint-level comparison charts

---

## 💡 Key Findings

1. **Feature engineering is the multiplier** — relative and normalised lap-time features cut baseline MAE from ~3.7 s to ~0.88 s with the same linear model
2. **Tyre degradation is nonlinear** — tree-based models substantially outperform linear baselines
3. **Gradient Boosting generalises better cross-race** than Random Forest, despite higher global MAE
4. **MLP > LSTM/GRU for pointwise prediction** — sequence models shine for full degradation curve forecasting (seq2seq), not single-value targets
5. **Stacking combines the best of all worlds** — interpretability (LR) + nonlinear correction (RF) + smoothing (MLP)

---

## 🚀 Future Work

- Sequence-to-sequence degradation prediction for full stint-curve outputs
- Incorporate **track temperature, weather, and Safety Car laps**
- **Pit window estimation** and strategy simulation tools
- **Uncertainty quantification** for strategy risk modelling

---

## 🛠️ Setup

```bash
pip install fastf1 pandas scikit-learn matplotlib seaborn tensorflow torch
```

All notebooks run on **Google Colab** (zero-install). Mount your Google Drive and run cells in order.

---

## 👩‍💻 Author

**Hana Tharwat Elzarka**  
BEng Computer & Communications Engineering — Alexandria University (graduating June 2026)  
[LinkedIn](https://linkedin.com/in/hana-elzarka-2871b7282) · [GitHub](https://github.com/hanaelzarka16)

---

*Built with FastF1, scikit-learn, TensorFlow, and a lot of race weekend data.*
