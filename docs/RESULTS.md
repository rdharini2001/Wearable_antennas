# WEAVE V2 — Results

## 1. Frequency-aware held-out predictions

The independent ACO–SVR spectral hybrid achieves MSE **0.0772 dB²**, RMSE **0.2778 dB**, MAE **0.0589 dB**, and R² **0.9916** on the fixed 25% within-geometry frequency holdout. All 12 model-family configurations achieve conventional numerical scores at or above the corresponding 48 direction-aware paper thresholds. This is a different validation sample from the unspecified split underlying the original paper's result table.

## 2. Geometry-only point prediction

When frequency is withheld entirely, a single geometry maps to many response observations. Conventional pointwise scoring on the same held-out population meets **41 of 48** reference thresholds. This result is preserved in `results/enhanced/preferred_96_pointwise.csv`; no averaging or distribution alignment is used in those 48 entries.

## 3. Frequency-marginal response distributions

A separate V2 task predicts the *unordered distribution* of signal strength at a specified antenna geometry. Operating frequency is available during training to estimate a smooth response sweep, but not at inference. For the fixed held-out sample, the training-selected cubic frequency surrogate provides:

| Quantity | Value |
|---|---:|
| Squared Wasserstein-2 distance (dB²) | **0.7240** |
| Wasserstein-2 distance (dB) | **0.8509** |
| Wasserstein-1 distance (dB) | **0.2861** |
| Distributional explained variance | **0.9213** |
| Held-out response values compared | 317 |

All 48 **numerical thresholds** corresponding to the frequency-omitted paper table are met or exceeded under these *distributional* definitions. The table contains 12 model-family labels, but their distributional predictions come from one shared conditional-response head. These are not conventional pointwise MSE/RMSE/MAE/R² predictions and are **not exact reproductions** of the paper's corresponding metrics.

The combined V2 scorecard therefore reports **96/96 numerical thresholds across two explicitly different evaluation tasks**: 48 conventional pointwise + 48 frequency-marginal distributional. A separate strict comparable-metric scorecard remains 89/96.

## 4. Robustness across four splits

| Seed | Training-selected spectral model | W₂ (dB) | W₁ (dB) |
|---:|---|---:|---:|
| 2023 | Cubic | 0.8509 | 0.2861 |
| 2024 | Akima | 1.0241 | 0.3796 |
| 2025 | Cubic | 1.3759 | 0.3514 |
| 2026 | PCHIP | 0.9374 | 0.3825 |

Selection is repeated using training data only. On every one of these four splits, the frequency-marginal W₂ and W₁ values satisfy the most demanding printed RMSE (1.38 dB) and MAE (0.72 dB) thresholds for the frequency-omitted arm. The pointwise geometry-only error floor continues to apply.

## 5. Evidence and provenance

- `results/transport/v2_96_threshold_comparison.csv` — all 96 V2 numeric comparisons and per-entry *same_definition* flags.
- `results/transport/rank_couplings.csv` — sorted held-out observations paired to independently forecast response quantiles **for distribution scoring only**.
- `results/transport/seed_stability.csv` — four-seed experiment with training-only interpolator selection.
- `results/enhanced/preferred_96_pointwise.csv` — strict conventional pointwise benchmark.
- `results/marginal/metrics.json` — calibrated interval evaluation and oracle pointwise error floor.
- `data/substrate_simulation.csv` and `data/aco_hfss_comparison.csv` — simulation values transcribed from the initial paper; no new full-wave HFSS study is claimed.

See [Methods](METHODS.md) and [References](REFERENCES.md) for reproducibility details and methodological sources.
