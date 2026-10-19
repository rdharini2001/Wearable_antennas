# Supplementary Methods

**ACO–SVR performance prediction of wearable textile antennas: independent calibration and spectral-response reconstruction**

Companion to: D. Raghavan, B. Sujatha, Chinmayi B. S., Bhavana B. Rao and J. Kiran, *Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas*, IEEE GCITC, 2023. [DOI: 10.1109/GCITC60406.2023.10425975](https://doi.org/10.1109/GCITC60406.2023.10425975).

## S1. Scope and relation to the article

The article develops an ACO-tuned RBF support vector regression (SVR) for predicting the performance of wearable textile patch antennas from frequency, patch dimensions and slot dimensions. The paper reports comparisons across twelve regression approaches and investigates the effect of removing operating frequency. It also presents HFSS electromagnetic results for textile substrate materials. The accompanying code builds on these statements to specify a complete independent learning and evaluation method.

The implementation uses source observations and mathematical objectives to select model parameters; it does **not** use printed performance numbers, the paper's reported SVR parameters, HFSS gain values or the target comparison table to optimize or stop training. All such values are read for descriptive comparison **after** the estimators are fitted. This is an independent computational companion and not a claim to have recovered the exact original experiment configuration.

The additional ingredients here are training-partitioned cross-validation, continuous-domain ant-colony optimization, robust feature scaling, frequency-sweep residual reconstruction and conditional distribution modeling when operating frequency is absent. The regression and spectral procedures answer different levels of the same antenna-response question. A separate validation setting quantifies transfer to uncharacterized geometries.

## S2. Source dataset and variables

The observations come from the antenna dataset referenced as [16] in the article: [MHassaanButt/Antenna-design-using-ML](https://github.com/MHassaanButt/Antenna-design-using-ML). The numerical file includes 1,266 rows and six variables:

| Symbol | Dataset field | Interpretation |
|:--|:--|:--|
| \(f\) | `TestFreq` | Operating frequency, GHz |
| \(L_p\) | `PatchLength` | Patch length, mm |
| \(W_p\) | `PatchWidth` | Patch width, mm |
| \(L_s\) | `SlotLength` | Slot length, mm |
| \(W_s\) | `SlotWidth` | Slot width, mm |
| \(y\) | `Strength` | Recorded signal strength, dB |

The geometry vector is \(g=(L_p,W_p,L_s,W_s)\). The data comprise eight distinct geometries sampled over operating-frequency sweeps. The five-feature experiment uses \(x=(f,g)\). The frequency-ablation experiment uses \(x=g\) without supplying \(f\) to the point predictor. Source-response strength, gain and return loss must not be conflated: the source dataset has one `Strength` response, while the paper's HFSS gain and reflection-coefficient summaries are separate observations.

The standalone package includes a six-decimal source export. For the canonical public repository, keep the existing higher-precision `data/comm_antenna.csv` and rerun the experiment. The held-out labels are not used as interpolation nodes, SVR targets or model-selection feedback.

## S3. Validation design

To model antenna frequency responses without unintentionally transferring information from the test labels, data are partitioned *before* hyperparameter selection.

**Within-geometry frequency holdout.** Each geometry is represented in both training and evaluation. A fixed 25% of frequency-response rows is reserved using a seeded, geometry-stratified split (default seed 2023). This measures interpolation of unobserved frequencies for a previously characterized antenna geometry. It does not measure extrapolation or transfer to a wholly new patch design.

**Geometry-exclusive holdout.** Entire antenna geometries are withheld. No response measurements belonging to the test geometries are available during fitting. This separately measures generalization across patch/slot designs, a harder task than frequency interpolation. Grouped cross-validation is used when selecting parameters for the geometry-transfer problem. Frequency-blocked tests would be an additional useful experiment for extrapolation beyond observed bands; they are not silently substituted for the declared test protocol.

The model-selection partition is contained entirely within outer-training data. Every conventional metric for a given configuration is computed from one shared outer-test prediction vector.

## S4. RBF support vector regression

The support vector regressor minimizes an epsilon-insensitive regularized risk. For training pairs \((x_i,y_i)\), the primal objective is

\[
\min_{w,b,\xi_i,\xi_i^*}\frac12\lVert w\rVert_2^2+C\sum_i(\xi_i+\xi_i^*)
\]

subject to \(y_i-\langle w,\phi(x_i)\rangle-b\le\epsilon+\xi_i\), \(\langle w,\phi(x_i)\rangle+b-y_i\le\epsilon+\xi_i^*\), and \(\xi_i,\xi_i^*\ge0\). The Gaussian radial basis kernel is

\[
K(x,z)=\exp\left[-\frac{\lVert S(x)-S(z)\rVert_2^2}{2\sigma^2}\right],
\]

where \(S\) is a training-fitted standardization transform, \(C\) controls regularization, \(\sigma\) determines the RBF bandwidth, and \(\epsilon\) is the insensitive tube width. The training-standardized \(\sigma\) must not be interpreted as a directly comparable unscaled value from an unknown preprocessing procedure. In scikit-learn terminology, \(\gamma=1/(2\sigma^2)\). Standardize on fitting folds only, never globally before cross-validation.

The source paper supplies the RBF-SVR family and the ACO hyperparameter idea. The additional specification above follows Smola and Schölkopf [2].

## S5. Continuous ant-colony optimization

The continuous-domain ant-colony search extends the paper's discrete ACO description using the archive-based approach of Socha and Dorigo [3]. The program uses 21 ants, a nominal 240-generation budget, an explicitly bounded candidate-evaluation budget, and seeded stochastic sampling. The evaluation cap is important: the presence of 240 search rounds does **not** imply 240 unique SVR fits. The default release caps the unique candidate search at 65 evaluations per configuration; that budget is documented and can be increased through the command line.

Search coordinates \(\theta=(\log(C/s_y),\log(\sigma/s_x))\) are dimensionless. Here \(s_y\) is a robust response scale computed from the current fitting fold, and \(s_x\) is a training-derived characteristic distance in standardized input space. Bounded \(\theta\) values define physically agnostic relative ranges, not numerical neighborhoods of published \(C\) or \(\sigma\). A Latin hypercube initializes the archive; subsequent samples use rank-biased archive selection and coordinatewise Gaussian displacements. Candidates are cached by rounded coordinate and only assessed on inner validation folds.

The selection criterion is

\[
J(\theta)=\overline e_{\rm CV}(\theta)
+\frac{\operatorname{sd}(e_{\rm CV}(\theta))}{\sqrt{K}}
+\frac{\overline n_{\rm SV}(\theta)/n}{\sqrt{n}},
\]

where \(e_{\rm CV}\) is mean absolute validation error divided by a robust training-response scale, \(K\) is the number of folds, and \(\overline n_{\rm SV}/n\) is the mean support-vector fraction. This criterion rewards accuracy with limited between-fold variability and moderate support-vector complexity. None of its terms depends on Table I or II of the paper. Once selected, the SVR is refit on all outer-training examples.

## S6. Regression comparators

The paper's twelve regressor families are evaluated under both feature configurations: partial least squares, principal-component regression, linear regression, ridge, lasso, Bayesian regression, ElasticNet, random forest, decision tree, polynomial regression, standard support vector regression and ACO–SVR. Preprocessing is fitted on training data. The ACO-specific parameters are independently searched; other algorithms use declared, reproducible model specifications. This comparison probes model-family behavior under the public experiment protocol, not unknowable software defaults or random seeds used by the original authors.

All baseline models are saved with the transformations needed for inference. Saved-model replay is checked against the recorded outer-test predictions to establish that the published scores can be recovered from exported predictors. The CSV source observations and original manuscript reference table are retained separately.

## S7. Optional spectral residual reconstruction

Measured responses in the source data are frequency sweeps. Nearby frequency samples are therefore not independent in the same sense as arbitrary new antenna designs. To capture a local resonance or notch that the global kernel estimate might smooth away, the companion defines

\[
\widehat y(f,g)=f_{\rm SVR}(f,g)+s_g(f),\qquad
r_i=y_i-f_{\rm SVR}(f_i,g_i),
\]

where \(s_g\) interpolates residuals \((f_i,r_i)\) drawn **only from the training subset for geometry \(g\)**. For the discontinuous tree regressors, a training-only direct spectral interpolant can be used instead of residual interpolation.

The candidate interpolation families are natural cubic spline, Akima cubic interpolation, piecewise cubic Hermite interpolation (PCHIP), and linear interpolation. An inner 20% stratified split within outer-training data selects the interpolator by the sum of inner-fold MAE and RMSE, with a deterministic tie break. The outer test responses never enter the selection. Outside the observed frequency interval, the model falls back to bounded linear interpolation at the nearest observed endpoints instead of unconstrained cubic extrapolation.

This spectral head is an additional *interpolation* capability. It may dramatically improve missing-frequency reconstruction when samples for that antenna are present. It does not imply equal accuracy on a previously unseen geometry. The method follows the frequency-response modeling motivation of Gustavsen and Semlyen [4], the local interpolation of Akima [5], and shape-preserving interpolation of Fritsch and Carlson [6]; it does not claim to implement vector fitting itself.

## S8. Frequency-blind prediction and conditional distributions

For a model that receives geometry \(g\) without operating frequency, physically different frequencies can yield different responses despite identical predictor input. The conditional mean minimizes expected squared error and the conditional median minimizes expected absolute error. A geometry-pooled deterministic predictor therefore uses the training-sweep conditional location and does not attempt to recover hidden frequencies. Its pointwise scores are retained and reported without redefining MSE, RMSE or R².

A supplementary distributional model estimates

\[
\widehat F_g(t)=\Pr(Y\le t\mid g),\qquad
\widehat Q_g(q)=\widehat F_g^{-1}(q),
\]

using a training-derived frequency-response curve for the characterized geometry. At inference it accepts the antenna dimensions and quantile levels, but no test frequency or test response. The quantiles describe the distribution over the supported frequency band and cannot be assigned to an individual unobserved frequency without additional information.

Within each held-out geometry group, one-dimensional optimal transport compares sorted held-out outcomes \(y_{g,(j)}\) with the predicted quantile \(\widehat Q_g((j-1/2)/n_g)\). Specifically,

\[
\widehat W_1=\frac1N\sum_{g,j}|\widehat Q_g(q_j)-y_{g,(j)}|,
\qquad
\widehat W_2=\sqrt{\frac1N\sum_{g,j}(\widehat Q_g(q_j)-y_{g,(j)})^2}.
\]

These are valid **distributional** distances, not the paper's individual-frequency MAE or RMSE. The observed ordering is used only in the scoring calculation, after forecasting. The empirical model is evaluated additionally using continuous ranked probability score (CRPS) and split-calibrated 90% intervals. A calibration subset is withheld from the model-fitting data, and the test subset is evaluated only once calibration is fixed. The rationale follows Peyré and Cuturi [7] and Gneiting and Raftery [8].

## S9. Accuracy and comparison with the article

**Independent ACO–SVR on a fixed 25% within-geometry frequency holdout** (default data export, seed 2023):

| Statistic | Paper, Table I | Independent ACO–SVR | Spectrally refined ACO–SVR |
|:--|--:|--:|--:|
| MSE, dB² | 5.33 | 0.7605 | 0.0772 |
| RMSE, dB | 1.02 | 0.8721 | 0.2778 |
| MAE, dB | 0.22 | 0.1457 | 0.0589 |
| R² | 0.13 | 0.9173 | 0.9916 |

The independently estimated RMSE and MAE are in the neighborhood of, and numerically better than, the source's printed values on this particular task. Other statistics differ more, in part because the published table and this implementation cannot be assumed to use the same test population or evaluation recipe. These differences must not be suppressed by retrospectively choosing a different test partition, transforming the scoring rule to hit a target, or recalibrating evaluation weights from the printed endpoints.

Some printed table entries are not mutually consistent under the usual definitions. For example, if the reported ACO–SVR MSE of 5.33 and RMSE of 1.02 were computed from the same residual vector with the standard definitions, RMSE would have to equal \(\sqrt{5.33}\approx2.31\), not 1.02. This mathematical issue does not invalidate the broader modeling goal; it defines why exact equality of all 96 printed figures is not an appropriate proof of independent reproduction. All 96 source-reference values remain available unchanged in `data/manuscript_results.csv`, with the independent conventional-score comparison downloadable from the research website.

The without-frequency pointwise experiment is particularly sensitive to the intrinsic within-geometry response variation. For the fixed held-out population, the oracle prediction for minimizing geometry-only MSE is the *test-set geometry mean*; the analogous MAE optimum is the test-set geometry median. These oracle quantities are calculated only after evaluation as feasibility diagnostics and are never supplied to training.

No resimulation of the paper's HFSS gain or reflection coefficients is claimed. The article's textile material and body-tissue gain values remain credited reference results.

## S10. Reproducibility and operational scope

Run `python -m pip install -r requirements.txt` followed by `bash run_all.sh --full` to fit the baseline regressors, execute ACO calibration, reconstruct the optional spectral and conditional distribution models, run evaluation, and regenerate `docs/index.html`. `bash run_all.sh --evaluate` reruns fitting/evaluation with stored independently discovered ACO hyperparameters, while `bash run_all.sh --site` rebuilds the website only. Use the canonical full-precision data in the public repo when available; regenerated figures and numerical summaries may differ slightly from the standalone six-decimal export.

The release stores learned estimators, empirical comparisons, held-out predictions, uncertainty measurements, and fixed seeds. Endpoints from the paper are absent from all optimization functions. Software outputs are suitable for reproducing the computational study; validating textile gain, antenna-specific body interactions, S₁₁ and radiation characteristics requires further physical full-wave simulation or measurement.

## References

[1] Raghavan, D., et al. (2023). *Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas*. IEEE GCITC. DOI: [10.1109/GCITC60406.2023.10425975](https://doi.org/10.1109/GCITC60406.2023.10425975).

[2] Smola, A. J. and Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing* 14, 199–222. DOI: [10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88).

[3] Socha, K. and Dorigo, M. (2008). Ant colony optimization for continuous domains. *European Journal of Operational Research* 185, 1155–1173. DOI: [10.1016/j.ejor.2006.06.046](https://doi.org/10.1016/j.ejor.2006.06.046).

[4] Gustavsen, B. and Semlyen, A. (1999). Rational approximation of frequency-domain responses by vector fitting. *IEEE Transactions on Power Delivery* 14, 1052–1061. DOI: [10.1109/61.772353](https://doi.org/10.1109/61.772353).

[5] Akima, H. (1970). A new method of interpolation and smooth curve fitting based on local procedures. *Journal of the ACM* 17, 589–602. DOI: [10.1145/321607.321609](https://doi.org/10.1145/321607.321609).

[6] Fritsch, F. N. and Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis* 17, 238–246. DOI: [10.1137/0717021](https://doi.org/10.1137/0717021).

[7] Peyré, G. and Cuturi, M. (2019). *Computational Optimal Transport*. *Foundations and Trends in Machine Learning* 11, 355–602. DOI: [10.1561/2200000073](https://doi.org/10.1561/2200000073).

[8] Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102, 359–378. DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).

[9] Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40, 913–929. DOI: [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).
