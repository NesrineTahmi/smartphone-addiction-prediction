# Predicting Smartphone Addiction

Kaggle Playground Series competition: predict the probability of smartphone addiction (`addicted_label`) for each `id`, evaluated on **ROC AUC**.

## Data

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Repo structure

```
├── data/                                  # raw competition files 
├── notebooks/
│   ├── EDA.ipynb                          # exploratory data analysis
│   ├── baseline_model.ipynb               # first LightGBM baseline
│   ├── ensemble_modeling.ipynb            # seed-averaged LightGBM ensemble
│   ├── blended_catboost_lightgbm.ipynb    # LightGBM + CatBoost blend
│   ├── lightgbm_oof.ipynb                 # full feature engineering + 5-fold OOF CV (best model)
│   └── catboost_info/                     # CatBoost training logs (auto-generatedd)
├── .gitignore
└── README.md
```

## Approach

- **Target**: `addicted_label`, binary, ~71% positive class.
- **Missing values**: up to ~19% missing on some columns; confirmed missingness is not informative on its own (target rate is flat whether a value is missing or present), so no missingness flags per column, but an aggregate *missing count* feature was still useful.
- **Model**: LightGBM, trained with stratified 5-fold cross-validation and out-of-fold (OOF) predictions for honest validation.
- **Feature engineering**:
  - Ratios and aggregates across the screen-time cluster (`social_ratio`, `gaming_ratio`, `work_ratio`, `weekend_vs_daily`, etc.)
  - Missingness counts (overall / numeric / categorical)
  - "CI" reconstruction features exploiting the synthetic dataset's underlying formula (`daily_screen_time_hours ≈ social + gaming + work`) to recover values when one component is missing
  - Frequency encoding (value counts across train+test, no target leakage)
  - Fold-safe target encoding (smoothed, computed strictly within each training fold to avoid leakage)

## Results

| Notebook | Approach | Public LB |
|---|---|---|
| `baseline_model.ipynb` | LightGBM baseline | 0.96416 |
| `ensemble_modeling.ipynb` | Seed-averaged LightGBM | 0.96468 |
| `blended_catboost_lightgbm.ipynb` | LightGBM + CatBoost blend | 0.96452 |
| `lightgbm_oof.ipynb` | LightGBM + full feature engineering + 5-fold OOF CV | 0.96981 |
