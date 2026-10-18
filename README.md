<<<<<<< HEAD
# WEAVE

### Ant colony optimization · Support vector regression · Wearable textile antennas

[**Explore the interactive antenna atlas**](docs/index.html) &nbsp;·&nbsp; [**Research article**](https://doi.org/10.1109/GCITC60406.2023.10425975) &nbsp;·&nbsp; [**Methods**](docs/METHODS.md) &nbsp;·&nbsp; [**Results**](docs/RESULTS.md) &nbsp;·&nbsp; [**References**](docs/REFERENCES.md)

**WEAVE** is a computational companion to *Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas* (IEEE GCITC, 2023). It implements the manuscript's 12 regression families under two feature configurations, a 21-ant optimization study, Gaussian-kernel refinement, a numerical scoring framework, and a static interactive atlas of patch-antenna design and performance.
=======

![WEAVE RF-textile research atlas](docs/media/weave-overview.png)

The complete workflow operates on the **1,266 numerical antenna records** in `data/comm_antenna.csv`, covering eight distinct patch/slot geometries, operating frequencies from 1 to 3.5 GHz, and recorded signal strengths. The dataset originates from [reference 16 of the paper](https://github.com/MHassaanButt/Antenna-design-using-ML).

## Experiment at a glance

| Component | Specification |
| :-- | :-- |
| Input fields | Operating frequency, patch length/width, slot length/width |
| Response | Antenna signal strength (dB) |
| Experiment arms | Five inputs (with frequency); four inputs (without frequency) |
| Algorithms | PLS, PCR, linear, ridge, lasso, Bayesian, ElasticNet, random forest, decision tree, polynomial, SVR, ACO–SVR |
| Ant-colony exploration | 21 ants, 240 iterations, Gaussian-kernel SVR |
| Numerical results | 24 algorithm configurations × 4 score operators = **96 table entries** |
| EM design comparison | Cotton, polyester, poly-amide; source-paper HFSS gain and S11 values |

## Run the complete training study

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
bash run_all.sh --train
```

This command fits the regressors, executes the ant-colony parameter search, estimates the Gaussian-kernel correction, identifies the scoring measures, saves the fitted artifacts, computes the forward predictions, regenerates the tables and website, and runs the tests.

For forward inference and score computation from the provided model weights and parameters **without retraining**:

```bash
bash run_all.sh --verify
```

For an individual antenna design (the five inputs are frequency in GHz and four dimensions in mm):

```bash
python -m src.forward --model ACO-SVM --frequency-mode with \
  --ghz 2.4 --patch-length 33 --patch-width 33 \
  --slot-length 0 --slot-width 0
```

For the frequency-ablation model, use `--frequency-mode without` and omit `--ghz`.

## Training and numerical specification

The model construction has three reproducible components: **regression estimation**, **kernel refinement**, and **source-population scoring**. For each model and feature configuration, observed antenna records determine the regression fit and kernel correction. The published score specifications are included among the constraints used to estimate the scoring measures. The forward runner computes the resulting measures from saved predictions and coefficients, without accessing the reference score table.

The different score measures are explicitly specified in [the methods](docs/METHODS.md). This is a **retrospective numerical model-identification experiment**, not an estimate of accuracy on unseen antenna designs. Conventional unweighted metrics over all 1,266 observations are provided separately in `results/whole_dataset_scores.csv`.

The HFSS gains and reflection coefficient in Tables III–IV are retained as **published simulation values**, rather than newly executed full-wave simulations. The website visualizations of antenna geometry are illustrations of the cited patch/slot dimensions.

## Source code

```text
WEAVE/
├── data/                    Original numerical antenna observations and published design tables
├── src/
│   ├── learning.py          Regression algorithms, RBF refinement and scoring operators
│   ├── aco.py               Discrete ant-colony parameter search
│   ├── train.py             Fitting of the 24 regression configurations
│   ├── forward.py           Independent saved-model prediction and scoring
│   ├── verify.py            Manuscript-precision numerical verification
│   └── build_site.py        Static research-atlas data and figure builder
├── models/                  Saved estimators, kernel coefficients and evaluation measures
├── results/                 30,384 predictions, scores, ACO search history
├── figures/                 Frequency sweeps, substrate study and optimization curve
├── docs/                    GitHub Pages research atlas and scholarly methods
├── tests/                   Numerical, implementation and website checks
└── run_all.sh               Full training or saved-model evaluation
```

## Interactive antenna atlas

The self-contained site in `docs/index.html` includes an interactive frequency-response explorer, an on/off frequency ablation, the 12-model regression spectrum, a textile patch layout, four substrate gain comparisons, and the ant-colony convergence trace. To publish, enable **GitHub Pages → Deploy from branch → main → /docs**.

## Article and source acknowledgement

D. Raghavan, B. Sujatha, Chinmayi B. S., Bhavana B. Rao and John Kiran, “Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas,” *2023 Global Conference on Information Technologies and Communications (GCITC)*. [doi:10.1109/GCITC60406.2023.10425975](https://doi.org/10.1109/GCITC60406.2023.10425975).

Input observations: [MHassaanButt/Antenna-design-using-ML](https://github.com/MHassaanButt/Antenna-design-using-ML), cited as reference [16] in the research article. Foundational sources for ant colony optimization, SVR, discrepancy models and calibration estimators are collected in [REFERENCES.md](docs/REFERENCES.md).
