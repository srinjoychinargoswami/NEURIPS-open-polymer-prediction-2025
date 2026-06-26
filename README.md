# NeurIPS Open Polymer Prediction 2025

Competition submission for the **NeurIPS Open Polymer Prediction 2025** Kaggle challenge.

**Competition Link:** https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025

---

## Overview

This project predicts **5 polymer physical properties** from molecular structure (SMILES strings):
- **FFV** (Fractional Free Volume) — 48% of competition metric weight
- **Tg** (Glass Transition Temperature)
- **Tc** (Critical Temperature)
- **Density**
- **Rg** (Radius of Gyration)

The challenge uses **weighted Mean Absolute Error (wMAE)** as the evaluation metric, with weights determined by property variability and dataset size across 2,240 competition participants.

---

## Results

**Placement:** 460th out of 2,240 participants (**79th percentile**)

**Competition Submission:** Version 58 (0.067 public leaderboard wMAE)

**Post-Competition Learning:** Version 76 (developed after competition ended) outperformed Version 58 on full test data, revealing critical overfitting to the limited public leaderboard. This experience demonstrated the importance of monitoring multiple metrics beyond validation MAE and understanding how models generalize beyond available evaluation windows.

Not a winning entry, but demonstrates solid multi-target regression fundamentals and ensemble methodology on a real Kaggle competition — with valuable lessons about avoiding overfitting to partial leaderboards.

---

## Methodology

The approach combines **target-specific feature engineering** with **version-based ensemble selection**. Each of the 5 targets receives optimized feature sets: Mordred 2D descriptors (~1,600 features), Morgan fingerprints (1,024 bits), MACCS fingerprints (166 bits), and hand-crafted polymer-specific features. For example, Tg uses 13 thermal-property descriptors (rotatable bonds, aromaticity ratio, functional group counts) to capture glass transition physics, while FFV uses base molecular descriptors only. Test predictions employ **four competing methods** (single best model, weighted ensemble, stacked ensemble, meta-learned ensemble) with automatic selection by validation MAE — ensuring each target uses its proven best approach.

The pipeline trains **three boosting models** per target (LightGBM, XGBoost, CatBoost) with version-specific hyperparameters: Tg/FFV use early stopping rounds of 100, while Tc/Density/Rg use 50 rounds with enhanced regularization. Meta-learned ensembles evaluate 14–15 candidate models (Ridge, SVR, RandomForest, MLP, etc.) via cross-validation, with sanity checking to reject suspiciously low scores. Data-driven clipping replaces hardcoded bounds using training data ranges (±5% margin), and all predictions are aligned to the same feature schemas across train/validation/test sets.

---

## Code Structure

- **Main notebook:** `.ipynb` file containing full training pipeline
- **Key stages:**
  - Data loading and combination (official + supplemental datasets)
  - Target-specific feature extraction (Mordred + fingerprints + domain features)
  - Parallel base model training (LightGBM, XGBoost, CatBoost)
  - Four-way method comparison (single, weighted, stacked, meta-learned)
  - Test prediction and submission.csv generation

---

## Environment & Dependencies

This code was developed in a **Kaggle notebook environment with restricted internet access**, requiring pre-packaged wheel (.whl) files for key dependencies:

- **RDKit** (cheminformatics) — installed via .whl
- **Mordred** (molecular descriptors) — installed via .whl
- **scikit-learn, pandas, numpy** (data science)
- **LightGBM, XGBoost, CatBoost** (gradient boosting)

The notebook includes custom wheel installation logic to handle the offline environment. Wheel files are referenced but not included in this repo; users with standard internet access can `pip install rdkit mordred` directly.

---

## Key Learnings

**The Overfitting Trap (Version 58 vs Version 76):** Version 58 achieved 0.067 wMAE on the public leaderboard and was submitted for competition ranking. However, Version 76, developed post-competition after seeing full test data performance, actually generalized better. This revealed critical overfitting: tuning specifically to optimize the partial public leaderboard led to worse generalization on unseen data. The insight forced a reckoning with model validation practices — relying solely on a single metric (validation MAE) can mask poor generalization. In real ML work, monitoring multiple evaluation windows, holdout sets, and cross-validation patterns is essential to distinguish between true improvement and leaderboard overfitting. This competition taught me to always ask: "Does this improvement generalize, or am I just fitting the evaluation set?"

## Key Features

- **Multi-target optimization** — each target trains independently with tuned hyperparameters
- **Ensemble diversity** — combines tree-based models with linear, SVM, and neural network meta-learners
- **Data-driven safety** — clipping bounds based on actual training data, not physical assumptions
- **Validation-driven selection** — all four ensemble methods compete; best selected automatically
- **Feature scaling** — target-specific approaches (Tg gets thermal features, Rg gets spatial features)

---

## Notes for Users

This is a **Kaggle competition submission**, not a production system. The code prioritizes competition metric (wMAE) over individual target MAE, and uses Kaggle-specific file paths (`/kaggle/input/`, `/kaggle/working/`). To adapt for local use:

1. Replace Kaggle paths with your own data directories
2. Install dependencies normally: `pip install rdkit mordred lightgbm xgboost catboost scikit-learn pandas numpy`
3. Update train/test CSV paths to match your file structure
4. Run cells sequentially; feature extraction is I/O intensive

---

## License

Apache License 2.0 — See LICENSE file for details.
