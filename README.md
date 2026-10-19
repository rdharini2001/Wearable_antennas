# WEAVE

### Ant colony optimization for wearable textile antenna performance prediction

[**Research website**](https://rdharini2001.github.io/Wearable_antennas/) · [**Supplementary methods (PDF)**](docs/SUPPLEMENTARY_METHODS.pdf) · [**Source paper**](https://doi.org/10.1109/GCITC60406.2023.10425975) · [**Numerical results**](docs/RESULTS.md) · [**References**](docs/REFERENCES.md)

![WEAVE — textile patch antenna and frequency-field visualization](docs/assets/antenna-atlas.svg)

WEAVE is the code and interactive research website associated with **“Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas”**, published at IEEE GCITC (2023). It investigates the relationship between patch/slot geometry, operating frequency and antenna signal strength with independently tuned ant-colony support vector regression. The computational framework includes training-selected spectral-response refinement and frequency-marginal prediction as documented methodological extensions.

The code implements the paper's central research problem, comparative regression families, frequency ablation and analytical patch dimensions. Its additional mathematical choices are derived in the [supplement](docs/SUPPLEMENTARY_METHODS.pdf). The original reference values are reserved for post-training comparison; they are not used as loss targets, calibration anchors or stopping conditions.

## Frequency-response prediction

The dataset contains 1,266 observations across eight wearable antenna geometries. The experiment below holds out 25% of frequencies within each represented geometry (seed 2023) and selects hyperparameters using training-only validation.

| Model | MSE (dB²) | RMSE (dB) | MAE (dB) | R² |
|:--|--:|--:|--:|--:|
| ACO–SVR | 0.7605 | **0.8721** | **0.1457** | 0.9173 |
| ACO–SVR with spectral refinement | 0.0772 | 0.2778 | 0.0589 | 0.9916 |
| Paper (reported Table I) | 5.33 | 1.02 | 0.22 | 0.13 |

The ACO–SVR baseline is close in scale to, and below, the paper's reported RMSE and MAE, using a different declared test partition. The spectral refinement addresses interpolation between measured frequencies of characterized antenna geometries and is not represented as generalization to completely new designs. All standard metrics are evaluated together on the same predictions. These results are an independently evaluated companion, not a claim that every printed number was recreated under an identical experimental protocol.

The four-feature, no-frequency ablation is evaluated both as a point predictor and, separately, as a conditional response distribution. Distributional Wasserstein and interval-calibration scores are not substituted for pointwise MAE, RMSE or R².

## Reproduce

Python 3.10+:

```bash
python -m pip install -r requirements.txt
bash run_all.sh --full
```

To rerun evaluation with independently selected ACO parameters or refresh the static research website:

```bash
bash run_all.sh --evaluate
bash run_all.sh --site
```

The repository includes the source dataset, trained prediction bundles, machine-readable numerical comparisons, tests, and the GitHub Pages website. The source `data/comm_antenna.csv` is attributed to [MHassaanButt/Antenna-design-using-ML](https://github.com/MHassaanButt/Antenna-design-using-ML), reference [16] of the paper.

## Project structure

| Location | Contents |
|:--|:--|
| `data/` | Antenna observations and article reference tables |
| `src/` | Independent ACO–SVR estimation, regression comparators, spectral and distributional models |
| `results/` | Saved predictors and empirical model evaluations |
| `docs/index.html` | The research website and interactive atlas |
| `docs/SUPPLEMENTARY_METHODS.pdf` | Mathematical supplement and evaluation definitions |
| `tests/` | Scientific and software-validation tests |

This research website can be hosted directly from **GitHub Pages → main branch → `/docs`**. The published gain and return-loss values attributed to HFSS are taken from the article, not claimed as newly simulated EM results.

**Citation:** D. Raghavan, B. Sujatha, Chinmayi B. S., Bhavana B. Rao, and J. Kiran, “Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas,” *2023 Global Conference on Information Technologies and Communications*, DOI [10.1109/GCITC60406.2023.10425975](https://doi.org/10.1109/GCITC60406.2023.10425975).
