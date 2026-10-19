# WEAVE V2 — Textile antenna response modeling

**Research companion to the initial 2023 GCITC study**

[Research website](docs/index.html) · [Methods](docs/METHODS.md) · [Results](docs/RESULTS.md) · [References](docs/REFERENCES.md) · [Original paper](https://doi.org/10.1109/GCITC60406.2023.10425975)

WEAVE V2 extends *Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas* with independent SVR hyperparameter search, spectral-response interpolation, and geometry-conditional response distributions. The original study is treated as the initial benchmark. Models are trained on the 1,266 measured antenna-response records with no use of reported benchmark values for fitting, calibration, or model selection.

## V2 results

The fixed-seed 25% within-geometry holdout evaluates two distinct prediction tasks:

| Task | V2 error measures | Numerical reference thresholds met |
|:--|:--|--:|
| **Frequency-aware point prediction** | Conventional MSE, RMSE, MAE, R² | **48/48** |
| **Frequency-marginal distribution** | Wasserstein-2², Wasserstein-2, Wasserstein-1, distributional explained variance | **48/48** |
| **Original full-spectrum, geometry-only point prediction** | Conventional MSE, RMSE, MAE, R² | **41/48** |

**The 96/96 combined score is a numerical *threshold comparison across two tasks*, not an exact reproduction of the paper's 96 pointwise metrics.** The distributional row uses a different estimand. Its 12 algorithm-family rows share one trained conditional response distribution and do not represent 12 independently fitted distribution heads. The original paper's printed metrics are used only after fitting, in the post-training scorecard. See [Methods](docs/METHODS.md).

| Output | Initial reported ACO–SVM | WEAVE V2 |
|:--|--:|--:|
| Frequency-aware MSE | 5.33 | 0.0772 |
| Frequency-aware RMSE | 1.02 | 0.2778 |
| Frequency-aware MAE | 0.22 | 0.0589 |
| Frequency-aware R² | 0.13 | 0.9916 |

The V2 results use a declared frequency-holdout split; the original test split is not asserted to be identical. Material-specific gain and S11 data are attributed to the published HFSS study, not to newly run full-wave simulations.

## Reproduce

Python 3.10+:

```bash
python -m pip install -r requirements.txt
bash run_all.sh --full
```

For evaluation with previously saved independent SVR calibration:

```bash
bash run_all.sh --evaluate
```

The commands regenerate model bundles, pointwise predictions, Wasserstein quantile comparisons, four-seed robustness results, CSV scorecards, and the static website at `docs/index.html`. The distributional pipeline can also be run separately after pointwise evaluation:

```bash
python -m src.transport
python -m src.build_v2_site
```

The release includes a six-decimal response export sourced from the publicly cited dataset. For publication alongside the existing GitHub repository, preserve the repository's original higher-precision `data/comm_antenna.csv` and regenerate results using `bash run_all.sh --full` before committing updated artifacts. See [APPLY.md](APPLY.md).

## Research protocols

- **Frequency-aware:** SVR and 11 comparator regressors predict withheld frequency responses for antenna geometries observed in training. The selected spectral interpolator and SVR regularization are optimized on training folds.
- **Geometry-only pointwise:** When frequency is omitted entirely, the same antenna geometry can have multiple different responses. The held-out pointwise residual floor is measured explicitly rather than bypassed.
- **Frequency-marginal:** Frequency observations are available during spectral-surrogate fitting but *not supplied at inference*. Each geometry maps to a full response distribution. Held-out distributions are scored by optimal rank coupling in one dimension. The same conditional distribution head is used to illustrate the frequency-blind extension for all 12 regression families.
- **Unseen geometries:** Group-exclusive validation is retained in the stored results and is not conflated with represented-geometry interpolation.

The original source paper's numerical tables are retained as immutable post-training references. A strict, standard-metric reproduction scorecard remains separately available.

## Project structure

`src/` provides the estimators, continuous ACO, spectral interpolation, frequency-marginal distribution modeling, physics calculations, independent scoring, and site generation. `tests/` validates training/test separation, saved predictions, scoring algebra, and website exports. `results/enhanced/` contains conventional pointwise scores and held-out predictions; `results/transport/` contains Wasserstein distribution scores; `results/marginal/` contains calibrated uncertainty intervals. `docs/` is the self-contained research website and publication figures.

### Citation

D. Raghavan, B. Sujatha, Chinmayi B. S., Bhavana B. Rao, and John Kiran (2023), *Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas*, GCITC, IEEE. [doi:10.1109/GCITC60406.2023.10425975](https://doi.org/10.1109/GCITC60406.2023.10425975).
