# sinter_plant_emmision_prediction
# 🏭 Sinter Plant Emission Prediction

Predicting air pollutant emissions from a steel sinter plant using machine learning — covering SO₂, NOₓ, CO, and PM across a 50,000-row process dataset with 11 operational inputs.

---

## 📋 Project Overview

Sinter plants are among the largest emission sources in integrated steel mills. Real-time prediction of flue gas pollutants (SO₂, NOₓ, CO, particulate matter) enables proactive emission control, reduces regulatory risk, and supports process optimisation — without the cost of continuous physical analysers on every output stream.

This project builds and evaluates a complete ML pipeline for that prediction task, culminating in a noise-ceiling analysis that reframes what "good performance" actually means for this dataset.

---

## 🎯 Targets

| Target | Description | Unit | R² Ceiling* |
|---|---|---|---|
| SO2_ppm | Sulphur dioxide | ppm | 0.696 |
| NOx_ppm | Nitrogen oxides | ppm | 0.321 |
| CO_ppm | Carbon monoxide | ppm | 0.279 |
| PM_mgNm3 | Particulate matter | mg/Nm³ | 0.676 |

\* *Theoretical maximum R² achievable on this dataset, computed from irreducible noise in the generation process (see Noise Analysis section below).*

---

## 📥 Input Features (11)

| Feature | Description |
|---|---|
| Coke_Breeze_Rate | Coke feed rate to the sinter bed |
| Moisture_Content | Moisture % of the feed mix |
| Ignition_Temperature | Burner ignition temperature (°C) |
| Bed_Temperature | Sinter bed temperature (°C) |
| Gas_Flow_Rate | Exhaust gas volumetric flow rate |
| Suction_Pressure | Under-grate suction pressure |
| Wind_Box_Pressure | Wind box pressure |
| Production_Rate | Sinter output rate (t/h) |
| Ambient_Temperature | Outside air temperature (°C) |
| Relative_Humidity | Ambient relative humidity (%) |
| Gas_Exhaust_Temperature | Flue gas temperature at stack |

---

## 🗂️ Dataset

- **Rows:** 50,000
- **Columns:** 15 (11 features + 4 targets)
- **Missing values:** ~0.5% per column — handled via median imputation
- **Source:** Synthetic dataset generated to reflect real sinter plant operating ranges
- **Kaggle:** [rajakumar26/sinter-emission](https://www.kaggle.com/datasets/rajakumar26/sinter-emission)

---

## 🔬 Key Finding: Noise Ceiling Analysis

A critical insight from this project — **the apparent "poor" R² for NOₓ and CO is a data property, not a modelling failure.**

After fitting a linear model to each target, we compute the residual noise ratio:


noise_ratio  = residual_std / total_std
R² ceiling   = 1 - noise_ratio²


| Target | Noise Ratio | R² Ceiling | Both models achieve |
|---|---|---|---|
| SO₂_ppm | 0.551 | **0.696** | ~99% of ceiling |
| NOₓ_ppm | 0.824 | **0.321** | ~98% of ceiling |
| CO_ppm | 0.849 | **0.279** | ~98% of ceiling |
| PM_mgNm³ | 0.569 | **0.676** | ~100% of ceiling |

NOₓ and CO were generated with ~82–85% irreducible random noise. No model — regardless of complexity — can exceed these ceilings on this dataset. Both final models are at or near ceiling across all four targets.

> **Lesson:** Always compute a data ceiling before benchmarking models. Labelling a model "Poor" because R²=0.30 looks low, when the ceiling itself is 0.32, is a misdiagnosis.

---

## ⚙️ Pipeline (v2)


STEP 1  →  Load & EDA
STEP 2  →  Preprocessing + Feature Engineering
STEP 3  →  Train / Val / Test Split (70 / 15 / 15)
STEP 4  →  Ridge Regression  (per-target alpha tuning)
STEP 5  →  XGBoost           (early stopping + per-target regularisation)
STEP 6  →  Cross-model comparison & ceiling-adjusted viability summary


### Feature Engineering (8 new features)

Physical domain knowledge was used to construct interaction terms:

| Feature | Rationale |
|---|---|
| Ign_x_Bed | Thermal NOₓ formation depends on the *product* of ignition and bed temperature |
| Temp_Ratio | Bed/ignition ratio — proxy for combustion completeness |
| Exhaust_Delta | Stack–bed temperature difference — heat loss indicator |
| Coke_x_Temp | Coke rate × bed temperature — combustion intensity |
| Flow_x_Pressure | Gas flow × suction pressure — SO₂ transport coupling |
| Coke_x_Pressure | Coke rate × suction pressure — sulphur mobilisation proxy |
| Flow_x_WindBox | Gas flow × wind box pressure — PM entrainment coupling |
| Humidity_x_Temp | Humidity × ambient temperature — moisture effect modifier |

### Model Configuration

**Ridge Regression**
- Alpha swept over 9 candidates (0.001 → 5000) per target via val MSE
- Trained on scaled engineered features (19 total)
- Selected because signal is predominantly linear for this feature set

**XGBoost**
- early_stopping_rounds=50 — stops at best validation iteration (~160–237)
- Per-target depth and regularisation: SO₂/PM use `max_depth=5`, NOₓ/CO use `max_depth=3` with `reg_lambda=10` (stronger regularisation for noisy targets)
- learning_rate=0.03`, `n_estimators=1000 (ceiling auto-detected by early stopping)

---

## 📊 Results (Test Set)

| Model | Target | R² | Ceiling | % of Ceiling | MAPE |
|---|---|---|---|---|---|
| Ridge | SO₂_ppm | 0.6943 | 0.696 | 99.8% | 7.60% |
| Ridge | NOₓ_ppm | 0.3155 | 0.321 | 98.3% | 5.70% |
| Ridge | CO_ppm | 0.2753 | 0.279 | 98.7% | 9.38% |
| Ridge | PM_mgNm³ | 0.6771 | 0.676 | 100.2% | 7.94% |
| XGBoost | SO₂_ppm | 0.6895 | 0.696 | 99.1% | 7.68% |
| XGBoost | NOₓ_ppm | 0.3123 | 0.321 | 97.3% | 5.72% |
| XGBoost | CO_ppm | 0.2745 | 0.279 | 98.4% | 9.40% |
| XGBoost | PM_mgNm³ | 0.6739 | 0.676 | 99.7% | 8.00% |

**Ridge slightly edges XGBoost** (avg R² 0.4906 vs 0.4876) — consistent with the finding that the underlying signal is primarily linear.

---

## 🚀 Getting Started

### Requirements


pip install pandas numpy matplotlib scikit-learn xgboost


### Run


# On Kaggle (default path)
# DATA_PATH is pre-set to the Kaggle input path in the script

# Locally — update DATA_PATH in the script header, then:
python sinter_pipeline_v2.py


### File Structure


├── sinter_pipeline_v2.py   # Complete single-file pipeline
├── README.md
└── data/
    └── sinter_50k_11inputs_4targets.csv   # Place dataset here if running locally


---

## 🔄 v1 → v2 Changes

| Issue in v1 | Fix in v2 |
|---|---|
| XGBoost: no early stopping → overfitting (gap 0.43) | early_stopping_rounds=50 added |
| XGBoost: max_depth=6 for all targets | Per-target depth (3 for NOₓ/CO, 5 for SO₂/PM) |
| All targets same regularisation | NOₓ/CO get reg_lambda=10 vs 3 for SO₂/PM |
| No feature engineering | 8 domain-informed interaction features added |
| Viability judged vs absolute R²=0.90 | Viability judged as % of data ceiling |
| 5 models (including RF, GBM, SVR) | 2 best models only (Ridge + XGBoost) |
| Alpha grid: 6 values | Alpha grid: 9 values (0.001 → 5000) |

---

## 🔭 Future Work

- **Regenerate dataset** with lower noise injection on NOₓ/CO (biggest single unlock — would raise ceilings to ~0.85)
- **Stacking ensemble** — Ridge predicts linear signal, XGBoost corrects residuals; meta-learner on val predictions
- **Polynomial Ridge** — degree-2 interaction terms for NOₓ/CO where signal is interaction-driven
- **Target-specific feature selection** — feed each XGBoost only its top correlated features to reduce noise dimensions
- **SHAP values** — for explainability and regulatory reporting

---

## 👤 Author

**Uplaksh Mittal**
B.Tech Civil Engineering 
Motilal Nehru National Institute of Technology Allahabad

---

