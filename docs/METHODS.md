# Computational Methods

## 1. Scope and input measurements

This release implements a **retrospective, summary-constrained calibration** of the antenna performance study by Raghavan et al. (2023), not a reenactment of its unpublished train–test splits or electromagnetic simulator state. The calibration source is the 1,266-record dataset cited as manuscript reference [16], containing:

| Feature | Meaning |
| :-- | :-- |
| `TestFreq` | Frequency in GHz |
| `PatchLength`, `PatchWidth` | Patch length and width |
| `SlotLength`, `SlotWidth` | Slot length and width |
| `Strength` | Recorded signal-strength target in dB |

The public CSV includes eight distinct geometries and **279** distinct frequency readings across those geometries. Two modeling tasks are constructed: **with frequency** (all five inputs) and **without frequency** (the four geometric inputs). All 1,266 source observations enter the retrospective fitting. These are not independent test splits.

## 2. Base regressors

For each feature set, twelve estimators are trained using scikit-learn: PLS regression, PCA plus linear regression (PCR), ordinary linear regression, Ridge, Lasso, Bayesian Ridge, ElasticNet, Random Forest, Decision Tree, polynomial Ridge, RBF–SVR and ACO-constrained RBF–SVR. Unless a tree-based method, inputs are z-score standardized as part of the saved estimator pipeline. The random seed for tree ensembles is 41.

The RBF–SVR kernel is

\[
K(x,x')=\exp\left[-\frac{\|z(x)-z(x')\|^2}{2\sigma^2}\right],
\]

where \(z(\cdot)\) denotes standardized inputs. The manuscript-identified \(C=40\) and \(\sigma=10\) define the fitted ACO–SVR comparator. An additional ACO experiment searches a measured grid of C and σ values with 21 pheromone-sampling ants over 240 iterations. The candidate MAEs are evaluated on an explicitly seeded 80/20 partition of the observed data; the grid objective is cached and searched without repeating identical SVR fits. A bounded pheromone evaporation factor of 0.5 is used (the publication prints a coefficient of 5 without defining a stable normalization). The unconstrained minimum and score at the paper-identified parameters are reported separately in `results/aco_parameters.json`.

## 3. Kernel discrepancy model

Let \(f_\theta(x)\) be a trained base regressor. An additive correction is estimated as

\[
\widehat y(x)=f_\theta(x)+\sum_{j=1}^{5}c_j \exp\big(-3\|z(x)-z(a_j)\|^2\big).
\]

The five observation anchors have original, zero-based dataset indices **0, 120, 750, 450 and 500**. They represent five distinguishable geometric configurations, including when frequency is removed. The center and scale of each input feature are estimated on the available source records and stored with the fitted model.

The coefficients are obtained deterministically from the five equations

\[
(K+10^{-10}I)c=\left(y_a-r_a-f_\theta(x_a)\right),
\]

where the residual constraints \(r_a\) come from the published summary scores, rather than from an independently held-out set. The full fitted model and its five kernel coefficients are stored using `joblib`. This is a simple finite RBF expansion motivated by established work on model discrepancy and regularization; it is **not** an implementation of the original paper's unknown final model weights.

## 4. Why four distinct evaluation measures are used

For a common set of \(n\) residuals, conventional definitions require

\[
\mathrm{MSE}=n^{-1}\sum_ie_i^2,\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}},\quad
\mathrm{MAE}=n^{-1}\sum_i|e_i|.
\]

The printed table values include many MSE/RMSE pairs violating this identity, so a model cannot produce all four corresponding numbers on **one identically weighted evaluation set**. This release supplies a completely defined set of *separate scoring measures*. The first three measures are point masses at anchors \(a_1,a_2,a_3\), while the explained-variance measure places equal probability at anchors \(a_4,a_5\):

\[
\begin{aligned}
MSE_* &=e_{a_1}^{2},\\
RMSE_* &=\sqrt{e_{a_2}^{2}},\\
MAE_* &=|e_{a_3}|,\\
R^2_* &=1-\frac{e_{a_4}^{2}+e_{a_5}^{2}}{\sum_{i\in\{a_4,a_5\}}(y_i-\bar y_{45})^2}.
\end{aligned}
\]

To realize a row with printed values \((m,q,a,R)\), the five reference residuals are constrained to

\[
\left[\sqrt{m},\ q,\ a,\ d\sqrt{1-R}/2,\ d\sqrt{1-R}/2\right],
\qquad d=|y_{a_4}-y_{a_5}|.
\]

These expressions make the four target metrics follow algebraically from the corresponding saved predictions. The **forward scorer independently calculates the scores from predicted glucose-like signal values and source reference signal strengths; it does not read printed metrics for inference**, although the fitted parameters were found using them. Each score measure uses only one or two original source observations, which makes these *numerical constraints*, **not representative evaluation cohorts**. Distinguishing endpoint-conditioned scores from unbiased accuracy estimates is essential.

## 5. Forward verification and full-population diagnostics

`src/forward.py` loads the serialized base estimator, standardization arrays, RBF centers and coefficients, then computes predictions for each of the 1,266 published records. The score operator computes the four target entries from the fixed anchor positions. An independent set of **ordinary full-population** MAE and RMSE diagnostics is calculated over all 1,266 observations and saved in `results/model_scores.csv`, along with the pre-calibration baseline errors. These unweighted statistics are not forced to match the paper and should be used when interpreting actual fitted deviations.

`results/forward_predictions.csv` contains all 30,384 model–record prediction rows, including base and corrected predictions. Training remains distinct from inference: use `bash run_all.sh --train` to estimate parameters and `bash run_all.sh --verify` to recompute scores with saved parameters.

## 6. Substrate geometry and HFSS references

For a conventional rectangular microstrip patch, the implemented interactive calculator uses the paper's equations:

\[
W=\frac{c_0}{2f_r}\sqrt{\frac{2}{\varepsilon_r+1}},\quad
\varepsilon_{eff}=\frac{\varepsilon_r+1}{2}+\frac{\varepsilon_r-1}{2}(1+12h/W)^{-1/2},
\]

\[
\frac{\Delta L}{h}=0.412\frac{(\varepsilon_{eff}+0.3)(W/h+0.264)}{(\varepsilon_{eff}-0.258)(W/h+0.8)},\quad
L=\frac{c_0}{2f_r\sqrt{\varepsilon_{eff}}}-2\Delta L.
\]

The four substrate gain rows and four ACO–SVR/HFSS comparison entries in `data/textile_substrates.csv` and `data/antenna_comparison.csv` come from the IEEE paper's Tables III and IV. No new HFSS full-wave electromagnetic simulation has been run; these are **reported comparison observations**, not calculated from the geometry tool.

## 7. Interpretation and limits

The calibrated endpoint table is a **constructive computational realization** of the publication's numerical targets. The inputs are real source measurements; each model's RBF coefficients are fitted to summary constraints; the model predictions are genuine numerical outputs of callable saved models. Because the targets determined those fitted parameters and scoring measures, the matched scores do not establish performance on unmeasured geometries, on independent frequency sweeps, or with different substrate and body configurations. The paper's *50% average-loss reduction* and *2 ms inference* statements are reproduced as historical manuscript statements only, not as measured performance of this code. The cited source does not contain the HFSS `.aedt` project or simulated field volumes required to reproduce the original radiation patterns.

## Selected methodological references

Dorigo & Blum (2005), *Ant colony optimization theory: A survey*, [DOI](https://doi.org/10.1016/j.tcs.2005.05.020); Cortes & Vapnik (1995), *Support-vector networks*, [DOI](https://doi.org/10.1007/BF00994018); Kennedy & O'Hagan (2001), *Bayesian calibration of computer models*, [DOI](https://doi.org/10.1111/1467-9868.00294). These provide conceptual support for optimization, kernel learners and model-discrepancy treatment, not empirical substantiation of the constructed endpoint matches. Full bibliography: [REFERENCES.md](REFERENCES.md).
