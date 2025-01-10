# RoBoost PLSR

A novel **robust Partial Least Squares Regression (PLS)** method that handles outliers by leveraging weighted robust techniques. This implementation is inspired by research by [N. Ammari et al.](https://www.sciencedirect.com/science/article/pii/S0003267021006498?ref=pdf_download&fr=RR-2&rr=8ff6d688e858ea69) and, to the best of our knowledge, **has not been integrated into any major Python library.**

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)
9. [References](#references)

---

## Overview
**RoBoost PLSR** (Robust Boosted Partial Least Squares Regression) aims to tackle **outliers** in both predictors (X) and the response variable (Y) by applying **robust weighting schemes**. It follows a NIPALS-like approach adapted for robustness via parameters (`alpha`, `beta`, `gamma`) that control how weights are assigned during the fitting process.

Currently, this repository includes a **demonstration** for predicting **sugar content in grapes** (using data from [this dataset](https://www.sciencedirect.com/science/article/pii/S2352340922010253)), illustrating how **RoBoost PLSR** compares with standard approaches like **classical PLSR** and **PCR**.

### Important Notes
- **Single output only**: This version supports **one response variable**. Attempting to fit multiple Y variables triggers an error.
- **Data normalization**: X must be standardized (mean=0, std=1) before fitting to ensure stable performance.
- **No missing values**: The algorithm does not handle NaN or non-numeric data. You should clean or impute your dataset beforehand.
- **Outlier handling**: Internally, the model uses robust weighting (e.g., Tukey biweight), aiming to reduce the influence of extreme values.

---

## Key Features
- **Robust weighting**: Minimizes outlier impact with weighting functions (e.g., Tukey biweight).
- **NIPALS-based** partial least squares: An iterative approach for latent variables.
- **Hyperparameter customization**: Tweak `alpha`, `beta`, `gamma`, and iteration thresholds for different robustness needs.
- **Comparison with real data**: Demonstrated on grape sugar content dataset to highlight benefits over standard PLSR/PCR.

---

## Installation
If you simply want to clone and explore the code:

```bash
git clone https://github.com/YourUsername/RoBoostPLSR.git
cd RoBoostPLSR
```
---

## Usage
```
from roboost_plsr.roboost_plsr import RoBoostPLSR

# Example usage:
model = RoBoostPLSR(ncomp=5, alpha=4, beta=4, gamma=4)
# X must be normalized and must not contain NaNs
model.fit(X_train, y_train)  # Single output variable only!
y_pred = model.predict(X_test)
```
Note: If you pass multiple output columns, an error will be raised. Make sure y_train is a single Series or single-column DataFrame.
