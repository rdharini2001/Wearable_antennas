# Numerical findings

Independent regression and antenna frequency-response modeling associated with the 2023 GCITC paper by Raghavan et al.

## Operating frequency supplied

All results below are computed from the same fixed 25% held-out response rows in the eight represented antenna geometries.

| Estimator | MSE (dB²) | RMSE (dB) | MAE (dB) | R² |
|:--|--:|--:|--:|--:|
| ACO–SVR | 0.7605 | 0.8721 | 0.1457 | 0.9173 |
| ACO–SVR with spectral response refinement | 0.0772 | 0.2778 | 0.0589 | 0.9916 |
| Source article, Table I | 5.33 | 1.02 | 0.22 | 0.13 |

The article and this implementation use different declared evaluation populations; the paper's partition is not independently established. Source values remain unaltered. The ACO-SVR baseline is the primary comparison; the refined result represents a specialized interpolation extension.

## Operating frequency omitted

A geometry-only point prediction cannot disambiguate responses at different frequencies of an identical geometry. Geometry-pooled ACO–SVR on the same frequency holdout has MSE 6.3152, RMSE 2.5130, MAE 1.0135, and R² 0.3135. A separate conditional distribution model gives held-out W1 = 0.2861 dB and W2 = 0.8509 dB. These last two are distributional distances, not individual-frequency prediction errors, and are not compared as if they were identical metrics.

## Electromagnetic reference

The research article reports HFSS gains of 6.8 dBi for cotton without body tissue, 9.6 dBi for cotton with body tissue, 4.7 dBi for polyester, and 5.9 dBi for poly-amide at 2.4 GHz. It also reports a cotton S11 comparison of -19.7 dB for ACO-SVR and -20 dB for HFSS. These are archival publication values rather than outputs of the current source-data regressors.

For data provenance, independent hyperparameter search, held-out protocols, regression and spectral equations, and complete bibliographic citations, see [Supplementary Methods](SUPPLEMENTARY_METHODS.md) or the [PDF](SUPPLEMENTARY_METHODS.pdf).
