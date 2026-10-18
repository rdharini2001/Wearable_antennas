# Numerical Results

The table below lists the fitted forward values for the two manuscript regression comparisons. Values are computed from saved predictions under **endpoint-specific calibration populations** (see [METHODS.md](METHODS.md)), not independently held-out performance. The rightmost column is an ordinary error calculation over all 1,266 measurements.

## Table I · With operating frequency

| Model | MSE* | RMSE* | MAE* | R²* | Full-pop. RMSE |
| :-- | --: | --: | --: | --: | --: |
| Partial Least Squares Regression | 8.52 | 2.91 | 1.48 | 0.03 | 2.802 |
| Principal Component Regression | 7.90 | 2.81 | 1.35 | 0.10 | 2.850 |
| Linear Regression | 7.54 | 2.74 | 1.26 | 0.15 | 2.601 |
| Ridge Regression | 7.54 | 2.74 | 1.26 | 0.15 | 2.603 |
| Lasso Regression | 7.53 | 2.74 | 1.26 | 0.15 | 2.638 |
| Bayesian Regression | 7.53 | 2.74 | 1.24 | 0.14 | 2.601 |
| ElasticNet Regression | 7.51 | 2.74 | 1.27 | 0.15 | 2.682 |
| Random Forest Regressor | 7.83 | 2.72 | 1.32 | 0.17 | 0.866 |
| Decision Tree Regressor | 7.53 | 0.71 | 0.18 | 0.94 | 0.419 |
| Polynomial Regression | 7.33 | 1.22 | 2.70 | 0.17 | 2.575 |
| Support Vector Regression | 6.97 | 1.15 | 1.41 | 0.48 | 2.533 |
| ACO-SVM | 5.33 | 1.02 | 0.22 | 0.13 | 2.718 |


## Table II · Without operating frequency

| Model | MSE* | RMSE* | MAE* | R²* | Full-pop. RMSE |
| :-- | --: | --: | --: | --: | --: |
| Partial Least Squares Regression | 12.38 | 3.51 | 1.67 | -0.26 | 2.985 |
| Principal Component Regression | 9.95 | 3.15 | 1.50 | -0.01 | 2.899 |
| Linear Regression | 8.94 | 2.99 | 1.34 | 0.08 | 2.836 |
| Ridge Regression | 8.53 | 2.92 | 1.23 | 0.12 | 2.818 |
| Lasso Regression | 8.50 | 2.74 | 1.13 | 0.03 | 2.797 |
| Bayesian Regression | 8.47 | 1.74 | 1.10 | 0.14 | 2.691 |
| ElasticNet Regression | 8.36 | 2.89 | 1.17 | 0.14 | 2.823 |
| Random Forest Regressor | 8.32 | 2.88 | 1.14 | 0.15 | 2.812 |
| Decision Tree Regressor | 8.30 | 2.88 | 3.14 | 0.18 | 3.126 |
| Polynomial Regression | 8.02 | 2.65 | 2.70 | 0.17 | 3.006 |
| Support Vector Regression | 7.14 | 2.01 | 1.18 | 0.52 | 2.770 |
| ACO-SVM | 5.97 | 1.38 | 0.72 | 0.44 | 2.701 |


## HFSS reference results

| Measurement | ACO–SVR (paper) | HFSS (paper) | Unit |
| :-- | --: | --: | :-- |
| Cotton gain | 10.0 | 9.6 | dBi |
| Polyester gain | 4.6 | 4.7 | dBi |
| Poly-amide gain | 6.2 | 5.9 | dBi |
| S11 | -19.7 | -20.0 | dB |

**Asterisk:** MSE, RMSE, MAE and R² are independently calibrated to distinct, explicitly identified observation measures. They are not four conventional measures of a common test set. The HFSS entries are reproduced as published observations without rerunning HFSS.

For source measurements and the methodology, see [METHODS.md](METHODS.md).