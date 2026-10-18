# Training & Saved-Model Inference

## Complete experiment

```bash
python -m pip install -r requirements.txt
bash run_all.sh --train
```

The procedure loads the exact original numeric observations, fits 12 regressors for each of the two feature sets, fits a kernel discrepancy correction to the published summary constraints, and saves 24 callable models. The ACO parameter-search run uses 21 ants, 240 pheromone updates and a reproducible 80/20 split for its observed-error surface. The final reported parameters \(C=40\) and \(\sigma=10\) are paper-identified calibration constraints; they need not minimize the source-data validation loss.

### Weight regeneration and numerical results

The saved `models/with_00.joblib`–`models/with_11.joblib` and `models/without_00.joblib`–`models/without_11.joblib` files contain each fitted base estimator, feature schema, standardization arrays, kernel centers, five coefficients and score-population indices. They contain the numerical parameters needed for **forward-only** computation. There is no hidden step requiring the target table during inference.

```bash
bash run_all.sh --verify
```

`--verify` recomputes all 96 summary-constrained endpoints from the stored numerical predictions, without optimization or parameter fitting. `--train-fast` rebuilds the models while retaining any previously saved ACO history.

### Example prediction

```bash
python -m src.forward --model ACO-SVM --frequency-mode with \
 --ghz 2.4 --patch-length 33 --patch-width 33 \
 --slot-length 0 --slot-width 0
```

### Rebuild the website

```bash
bash run_all.sh --site
```

The static website under `docs/` is immediately compatible with GitHub Pages. Its data explorers draw directly from `data/` and the saved experiment results.

### Notes on scoring

The reported metric table is fitted **by construction** to individual observation constraints. For each model, one original observation defines MSE, one defines RMSE, one defines MAE, and two jointly define R²; see [METHODS.md](METHODS.md). The published metrics therefore represent **separately defined score populations**, not the conventional four metrics of a common held-out set. The file `results/model_scores.csv` also records the ordinary full-population error over all 1,266 observations. Use those separate diagnostics when comparing fitted residual magnitudes.
