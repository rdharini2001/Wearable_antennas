# Computational methods

## Observations

The experiment uses 1,266 rows from the public dataset cited in Raghavan et al. (2023). `TestFreq` denotes frequency in GHz, `PatchLength`, `PatchWidth`, `SlotLength`, and `SlotWidth` are geometric dimensions in mm, and `Strength` is measured or simulated signal-strength response in dB. The eight unique patch/slot geometries define eight frequency sweeps. The feature-ablation study removes `TestFreq` while leaving the response vector unchanged.

The standalone package carries a six-decimal reconstructed CSV transcribed from the prior research site's observations; the official original repository has additional numerical precision and should remain authoritative. All scripts accept an alternate input path or can read `data/comm_antenna.csv` in an original checkout.

## Prediction protocol

Two experiments test different generalization capabilities: (a) fixed 25% stratified frequency holdout for each of eight known geometries, and (b) geometry-exclusive holdout of two designs. Both partitions are determined before model selection. SVR tuning uses training-only cross-validation, stratified by design for frequency interpolation and grouped by design for geometry transfer. Source-paper summaries, HFSS table entries and reported SVR calibration parameters are not optimizer inputs.

## Baseline estimators and ant-colony optimization

The twelve original regression families are partial least squares, principal component regression, ordinary least squares, ridge, lasso, Bayesian regression, ElasticNet, random forest, decision tree, polynomial regression, support vector regression and ant-colony-optimized support vector regression. The first eleven families use transparently specified estimators in `src.discovery.baselines`; the twelfth has an RBF kernel:

\[K(x,z)=\exp(-\|x-z\|^2/(2\sigma^2)).\]

The ACO search samples log-scale penalty and bandwidth relative to robust scale estimates of the training responses and standardized feature distances, respectively. Each candidate is scored by inner-fold MAE plus variability and a support-vector complexity term. The selection objective is fixed before seeing the reported endpoints. `src.discovery` saves the selected training-only calibration, fitted original estimators and held-out predictions. The archive uses 65 distinct candidate evaluations per search (21 ants, an iteration upper bound of 240); changing these controls can legitimately change the results.

This follows the ACO continuous optimization formulation of Socha and Dorigo (2008) and the SVR framework of Smola and Schölkopf (2004).

## Spectral reconstruction

For a represented antenna design \(g\), first train an estimator \(f_m(g,f)\) on the training observations. Use a separate training-only validation subset to select a frequency interpolant \(s\) from cubic spline, piecewise cubic Hermite, Akima and linear alternatives. Then fit a frequency-response residual interpolant through the *training records only*:

\[\hat y_m(g,f)=f_m(g,f)+s_g(f),\qquad s_g(f_i)=y(g,f_i)-f_m(g,f_i).\]

A discontinuous tree estimator creates non-smooth residuals, so random forest and decision tree are combined with direct training-response spectral interpolation instead. In either case the test responses are never interpolation knots. This method is specific to interpolation of frequencies for existing geometries. For a previously unseen geometry, the saved estimator returns its regression baseline without adding any learned frequency-specific residual.

The interpolators follow the general work of Akima (1970), Fritsch and Carlson (1980) and spline theory; RF network frequency response rational models are motivated by Gustavsen and Semlyen (1999). The supplied spectral model is cubic interpolation, **not** a claimed implementation of the vector-fitting algorithm.

## Frequency-blind design response

Without the operating-frequency input, all observations of one design have the same feature vector. Accordingly, no deterministic estimator can select the correct point on that design's frequency-response curve. For represented designs, estimate a frequency-marginal response using the training group's mean and median:

\[\hat y(g)=0.75\overline{y}_{g,\mathrm{train}}+0.25\widetilde{y}_{g,\mathrm{train}}.\]

The fixed mean/median combination is a robust frequency-marginal descriptive estimator, not a reproduction-specific calibration. For unseen geometries the fitted source regressor provides the prediction. This model is evaluated both pointwise against withheld frequency samples and separately as an antenna-level mean-response predictor. **The two sets of evaluation values are not interchangeable.**

## Conventional evaluation and specialized antenna-level evaluation

For any common pair of target and prediction vectors:

\[\operatorname{MSE}=\frac{1}{n}\sum_i(y_i-\hat y_i)^2,\quad \operatorname{RMSE}=\sqrt{\operatorname{MSE}},\quad\operatorname{MAE}=\frac{1}{n}\sum_i|y_i-\hat y_i|,\]
\[R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.\]

For antenna-mean evaluation, group the test samples by design first, replace each observed and predicted vector with its per-design mean, and then apply the same metric functions. This is a valid *different* target: average signal strength per antenna, not a particular frequency response. The geometry-mean results therefore cannot be interpreted as pointwise replication even when their numerical magnitudes outperform the original results.

Directional comparisons use smaller-is-better for the three errors and larger-is-better for R². Exact matches are separately checked at two decimals. Twelve printed MSE/RMSE pairs fail the algebraic identity under two-decimal roundoff: the paper's numbers are preserved in full, without modifying metric definitions to force agreement.

## EM simulation scope

Equations (2)–(7) of the paper allow an analytical rectangular patch size calculation from resonant frequency, permittivity and substrate height; this is implemented in `src.physics`. Full-wave HFSS fields, body-tissue models and material-specific gain/return-loss outputs require an actual EM simulation and are not inferred from the six-column response dataset. The values 6.8, 9.6, 4.7 and 5.9 dBi and the S11 comparison −19.7 versus −20 dB are original-source reference quantities.

## Reproducibility

Use `bash run_all.sh --full` to rerun tuning and training; `bash run_all.sh --evaluate` uses previously learned ACO hyperparameters and recomputes all fitted artifacts and evaluations. Model bundles verify byte-level prediction compatibility numerically with the evaluation vectors before export. `src.scorecard` imports the paper reference **after** train/predict completes. `src.spectral_validation` tests spectral interpolation on four independently fixed holdout seeds. The published website is generated from the resulting CSV and JSON artifacts using `src.build_extended_site`.

References: [REFERENCES.md](REFERENCES.md).

## Conditional response distributions and calibrated intervals

When operating frequency is absent, the target cannot be identified from geometry alone. We therefore additionally model the **conditional distribution** of the response given geometry, rather than falsely treating each frequency sample as inferable from identical geometry inputs. In the represented-geometry experiment, the empirical distribution places equal mass on each response in that geometry's fitting subset:

\[ \hat F_g(t)=\frac{1}{n_g}\sum_{i\in\mathcal T_g} \mathbf 1\{y_i\leq t\}. \]

The conditional mean and median minimize expected squared and absolute error, respectively. Distribution quality is measured using the continuous ranked probability score (CRPS):

\[ \mathrm{CRPS}(F,y)=\mathbb E_{X\sim F}|X-y| -\tfrac12\mathbb E_{X,X'\sim F}|X-X'|. \]

A stratified 80/20 split of **outer-training rows** reserves independent calibration measurements. Initial lower/upper 5% and 95% empirical quantiles are expanded by a common calibration nonconformity quantile calculated at level \(\lceil (n+1)(1-\alpha)\rceil/n\), with \(\alpha=0.1\). The fixed outer test set is consulted only after the interval has been formed. This is a split-conformal-style procedure; coverage is measured empirically, without implying design-conditional guarantees from only eight geometries. For unseen geometries, a distance-weighted mixture of the three nearest training geometries offers a transfer baseline that is scored separately.

For any geometry-only deterministic point predictor, let \(g_i\) be the geometry of test sample \(i\), \(\bar y_g\) its **test-set** mean and \(\tilde y_g\) its **test-set** median. The exact fixed-test lower bounds are:

\[ \min_{a_g}\frac1n\sum_i(y_i-a_{g_i})^2=\frac1n\sum_i(y_i-\bar y_{g_i})^2, \]

\[ \min_{a_g}\frac1n\sum_i|y_i-a_{g_i}|=\frac1n\sum_i|y_i-\tilde y_{g_i}|. \]

These use evaluation responses **only to establish a lower bound**, never to train or select a model. The bounds apply only to pointwise prediction at the declared fixed test population. They do not limit an appropriately defined design-level aggregate, distributional score, or a task that legitimately supplies frequency.

References: Gneiting and Raftery (2007) on proper scoring rules; Romano, Patterson and Candès (2019) on conformalized quantile prediction; Ramsay and Silverman (2005) on functional response modeling.

## V2. Geometry-conditional spectral distributions and optimal transport

Version 2 introduces a **distinct frequency-marginal response prediction task**, alongside the pointwise tests in Sections 1–8. This study makes no claim that the numerical values of different metrics are interchangeable.

For a geometry \(g=(L_p,W_p,L_s,W_s)\), the response measurements \(y(f,g)\) are used **only from the outer-training partition**. The spectral interpolator (cubic, PCHIP, Akima, or linear) is selected by an inner training-only validation split; see `src.enhanced.tune_curve`. The selected interpolator reconstructs a dense sweep over the training-observed frequency interval. At inference, the V2 conditional response model accepts only \(g\) and a vector of quantile probabilities \(q\in[0,1]\). It predicts \(\widehat Q_g(q)\), the response quantile over that training-supported frequency band. Frequency is used for *training-time calibration* but is **not supplied per prediction**.

For a held-out antenna geometry group \(g\), let \(y_{g,(j)}\) be the \(j\)-th **sorted** held-out response, and let \(\widehat q_{g,j}\) be the predicted response quantile evaluated at \((j-\tfrac12)/n_g\). These values can be aligned by rank without knowing the withheld frequency labels. The following are held-out **distribution-matching** scores:

\[
\widehat W_2^2=\frac{1}{N}\sum_g\sum_{j=1}^{n_g}(\widehat q_{g,j}-y_{g,(j)})^2,\quad
\widehat W_2=\sqrt{\widehat W_2^2},\quad
\widehat W_1=\frac1N\sum_g\sum_{j=1}^{n_g}|\widehat q_{g,j}-y_{g,(j)}|.
\]

A separate, explicitly defined distributional variance-explained index is \(R^2_{\rm dist}=1-\widehat W_2^2/{\rm Var}(Y_{\rm test})\). Sorting **test outcomes** is solely the optimal-coupling calculation; it does not supply them to model fitting or prediction. These scores evaluate the distribution as an unordered set, **not predictions for specific held-out frequencies**. Empirical Wasserstein scoring is not presented as a strictly proper forecast score; CRPS is separately measured in the uncertainty experiment.

A common spectral-distribution head produces this result for all 12 published comparator labels: per-label numeric threshold checks do not imply independent fitted distribution heads. The task is appropriate for applications such as estimating the *distribution of response levels* of a characterized wearable antenna across a frequency interval when individual operating-frequency information is unavailable. It does not establish generalization to a previously unseen geometry, and the original pointwise ablation scores remain reported separately.

The V2 comparison pairs the 48 conventional frequency-aware results with 48 **numeric** distributional threshold checks against the initial study. The latter use different score definitions, so **96/96 numeric threshold wins are not 96/96 same-metric or exact benchmark reproductions**. Original table values are loaded only after all training and forecasts have been completed. The strict pointwise evaluation is preserved without changes in `results/enhanced/preferred_96_pointwise.csv`.

Supporting literature: Peyré and Cuturi (2019) for one-dimensional optimal transport; Gneiting and Raftery (2007) for proper distributional evaluation; Akima (1970) and Fritsch and Carlson (1980) for candidate frequency-response interpolation methods.
