# Antenna response prediction results

The regression experiment characterizes the dependence of wearable textile antenna response on operating frequency and patch/slot geometry. The numerical dataset contains 1,266 sampled responses from eight antenna designs.

## Frequency-resolved regression

A geometry-stratified 25% frequency holdout reserves 317 measurements for evaluation. The fitting subset includes observations from each represented geometry, allowing interpolation to previously unobserved frequencies of characterized antenna configurations.

| Estimator | MSE (dB²) | RMSE (dB) | MAE (dB) | R² |
|:--|--:|--:|--:|--:|
| ACO–SVR | 0.7605 | 0.8721 | 0.1457 | 0.9173 |
| ACO–SVR with spectral reconstruction | 0.0772 | 0.2778 | 0.0589 | 0.9916 |

The complete twelve-family comparison in both feature settings is available as [regression metrics](downloads/regression_metrics.csv). The frequency-aware kernel estimator describes the global response relationship, while the spectral component interpolates locally observed resonant structure.

## Geometry-conditioned prediction

When the operating frequency is omitted, a geometry-only point estimator summarizes the response across the sampled frequency band. The geometry-pooled ACO–SVR model returns an MSE of 6.3152 dB², RMSE of 2.5130 dB, MAE of 1.0135 dB and R² of 0.3135 on the same test partition.

A frequency-marginal distribution model characterizes the response range using predicted quantiles rather than a single point response:

| Distributional measure | Value |
|:--|--:|
| Wasserstein-1 distance | 0.2861 dB |
| Wasserstein-2 distance | 0.8509 dB |
| Distributional explained variance | 0.9213 |
| Continuous ranked probability score | 0.7355 dB |
| Coverage of nominal 90% intervals | 93.1% |
| Mean interval width | 4.58 dB |

Pointwise and distributional results describe different but complementary prediction objectives. The former pairs a response with a specified input frequency; the latter summarizes response variation over the sampled operating band.

## Electromagnetic characterization

The textile substrate analysis considers cotton, polyester and poly-amide at 2.4 GHz. The HFSS gain values are 6.8 dBi for cotton in air, 9.6 dBi for cotton near body tissue, 4.7 dBi for polyester and 5.9 dBi for poly-amide. These simulation observations accompany the analytical patch design relations and the supervised response-regression dataset.

The detailed model formulation, evaluation protocols, derivations and bibliography are collected in [Supplementary Information](SUPPLEMENTARY_METHODS.pdf).
