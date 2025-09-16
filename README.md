# GDSC-NSUT Machine Learning Competition 2025 — CORRUCYSTIC_DENSITY

Predicting **CORRUCYSTIC_DENSITY** for the Kaggle competition:  
https://www.kaggle.com/competitions/recruitment-task-for-gdsc-ml

**Result:** Leaderboard — **5ᵗʰ place** (metric: **RMSE**)

---

## Overview

In the late 2060s, oceanic salvage drones dredged up black, coral-like growths with shimmering veins—fragments believed to be from an alien distributed biocomputer network, nicknamed **CORRUCYSTS**.  
The goal is to estimate **CORRUCYSTIC_DENSITY**, a scalar proxy for integrity/stability within possible Obeski neural architectures.

---

## Dataset

This dataset is a synthetic competition dataset designed to test feature-cleaning and modeling skills rather than domain knowledge. Feature names are deliberately obfuscated (e.g., `Z~x0<k`, `hp!`, `@wnsk>R`), so only statistical patterns matter. The target `CORRUCYSTIC_DENSITY` is a noisy continuous regression label with some invalid negatives, heavy tails, and systematic missingness (~8–10%) across many columns. Features fall into several groups: z-scored variables, probability-like [0–1] features, metadata/index-style non-predictive fields, and duplicated/redundant pairs.

## Problems Observed

- **Target column issues**
   - Missing labels in train; a small subset had **Negatives** (for a “density”).
   - Heavy-tailed distribution with outliers impacting RMSE.

- **Widespread missingness in features**
   - Many columns with **8–16%** missing values, often systematic.

- **Duplicate / redundant features**
   - Multiple near-duplicates increasing collinearity and variance.
   - duplicate_column_pairs = [ ("+U@", "A>."),("|G}", "14W$Q"),("ZZw3=!t", "<!!"),(".b6nl", "Kj,"),("fPqsI", "3I\\y"),("ZVf", "Jv[i]")]

- **Categorical noise**
   - 3 object/categorical columns with high missingness and **no measurable signal**.

- **Low-signal “probability-like” columns**
   - Bounded in `[0,1]`, mean ≈ 0.5, weak correlations—behaved like distractors.

- **Meaningless column names**
   - Most features have cryptic names like `Z~x0<k`, `hp!`, `@wnsk>R`.
   - No semantic meaning is available, so **domain understanding is blocked**, and only statistical/ML-driven feature selection is possible.

- **Feature scaling inconsistencies**
   - Many features (e.g., `vzo."`, `hp!`, `@wnsk>R`) look like **z-scored variables** (mean ≈ 0, std ≈ 1, range ≈ −3.5 to +4).
   - Others (e.g., `Z~x0<k`, `+U@`, `A>.`, `>?64:`) are **bounded between 0 and 1** with mean ≈ 0.5, suggesting probability-like or normalized values.
   - The dataset mixes scales heavily, which can confuse models.

## What I Did (Fixes Applied)

- **Targets**
  - Dropped rows with **missing target** in train (and mirrored for consistency in local test).

- **Redundancy & noise**
  - Identified **12 duplicates**; used one representative to impute, then **dropped 6** post-imputation.
  - Removed low-signal “probability-like” columns:  
    `Z~x0<k, >?64:, U"r, TSWm, w-u:jN'qI, PZ8, jNhEum`
  - Evaluated additional suspects one-by-one (`fPqsI, &%)LTaWRb, r2Ng, v0rt3X, b1oRb13`) → removed where they added noise and no lift.

- **Categoricals**
  - Imputed with `"MISSING"`, A/B tested impact (none) → **dropped all 3** categorical columns.

- **Metadata**
  - Excluded index/metadata fields from modeling.

**Result:** a **leaner, higher-signal feature set**, faster training, and more stable validation.

---


## Approach

**Modeling strategy:** three CatBoost regressors specialized by target range + simple averaging.

- **Bottom model** — tuned to do best on the **lowest deciles** of the target.
- **Top model** — tuned to do best on the **highest deciles** of the target.
- **Mid model** — generalist for the **20–80%** middle; hyperparameters selected via **Weights & Biases (W&B)** sweeps.

**Ensembling:**  
Attempted a meta-ensemble with **DESRegression** (`deslib`) over `[model_top, model_bot, mid]`.  
Due to a runtime error on `predict(X_valid)` close to deadline, final submission used a **simple unweighted average** of the three models (robust and effective given the complementary error profiles).

---

## Validation (decile RMSE diagnostics)

RMSE on validation by **true-y deciles** (lower is better). These illustrate the intended specialization of each model.

### Bottom model
- Summary: bottom tail strong, top tail weak; middle is moderate.
- Aggregates(*RMSE*): bottom 10% **42.17** · bottom 20% **64.85** · mid 20–80% **318.20** · top 20% **540.51** · top 10% **577.68**


### Top model
- Summary: mirror image of bottom model—excellent on high tail, poor on low tail.
- Aggregates(*RMSE*): bottom 10% **568.32** · bottom 20% **526.98** · mid 20–80% **290.17** · top 20% **53.99** · top 10% **42.52**

### Mid model
- Summary: balanced in the middle; reasonable on both tails.
- Aggregates(*RMSE*): bottom 10% **272.32** · bottom 20% **234.81** · mid 20–80% **109.45** · top 20% **231.90** · top 10% **265.48**


**Takeaway:** Averaging the three models reduces brittleness on extremes while maintaining strong performance in the center.

---





