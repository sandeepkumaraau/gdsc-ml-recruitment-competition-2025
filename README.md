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

The competition was with a synthetic dataset (randomized feature/label names; 8–10% missing per column), I used YData Profiling to audit feature quality and detect redundancy. 
I found 6 duplicate columns and 8 non-informative features; I kept one canonical column, used its duplicates to impute missing values, and dropped 14/46 redundant features. 
This reduced noise, improved validation performance, and sped up model convergence.



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





