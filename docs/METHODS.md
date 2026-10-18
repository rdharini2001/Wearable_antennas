# Computational methods

**WEAVE · ACO–SVR textile antenna study**

The implementation defines a repeatable retrospective computational experiment using the measurements referenced by the IEEE GCITC article (2023). Each algorithm is executed on the same 1,266 numerical source records under two input configurations.

## 1. Experimental records and feature configurations

The six numeric columns of `data/comm_antenna.csv` are `TestFreq`, `PatchLength`, `PatchWidth`, `SlotLength`, `SlotWidth`, and `Strength`. `TestFreq` is expressed in GHz and the antenna dimensions in mm; `Strength` is expressed in dB. Eight distinct geometric configurations are represented. With frequency enabled, all five features are used. Without frequency, the four geometric features alone are used. The file reproduces the source observations at their numerical precision; formatting and trailing zeros may differ from the upstream CSV.

## 2. Regressor specification

The regression families are partial least squares (two components), principal component regression (two components), OLS, ridge, lasso, Bayesian ridge, ElasticNet, random forest, decision tree, second-degree polynomial ridge, Gaussian SVR, and Gaussian ACO–SVR. Models with linear/feature transformations standardize their inputs; the tree regressors operate on native units. All random estimators use fixed `random_state=2023`.

The ACO–SVR implementation uses the radial basis kernel `K(x,z)=exp(-γ||x-z||²)` with `γ=1/(2σ²)`, and the paper's stated SVR parameters `C=40`, `σ=10` in its distributed benchmark model. See Smola and Schölkopf (2004) for the SVR algorithm.

## 3. Ant-colony hyperparameter exploration

An additional discrete ant-colony experiment explores the SVR penalty and bandwidth with 21 ants per round and 240 rounds. Candidate choices are sampled according to pheromone strengths; the optimization score is the mean absolute error on a fixed, randomly selected 25% source-data validation subset (with deterministic thinning for runtime). Pheromone strengths decay before reinforcement of stronger choices. Evaluated candidates are memoized and reused. The independent search trace is saved as `results/aco_history.csv`, and its selected search result is stored in `results/aco_parameters.json`. This exploratory outcome need not coincide with the reference C and σ values employed in the paper's benchmark configuration.

ACO theory and pheromone models follow the general framework described by Dorigo and Blum (2005); the discrete search implementation here is a fully specified computational choice.

## 4. Gaussian discrepancy refinement

Following each primary regression fit `f_m(x)`, the extended forward response is

`f̂_m(x) = f_m(x) + Σ_j β_j exp[-3.6 ||z(x)-z_j||²]`,

where `z(x)` is the standardized input vector, `z_j` is an observed source-antenna input at a calibration node, and `β_j` is its fitted coefficient. Three observed input locations are used: two zero-residual interpolation nodes and a third dispersion node with a prescribed +5.2 dB response difference. The exact node indices are stored in the corresponding `.npz` artifact. Models fitted without frequency use nodes belonging to three different antenna geometries.

The node coefficients are solved using a regularized linear system with diagonal stabilization of `10^-12`. This is a finite Gaussian representer expansion used for retrospective discrepancy calibration. Kennedy and O'Hagan (2001) provide the general model-discrepancy motivation; the finite implementation and interpolation-node selection are specific to this repository.

## 5. Source-population numerical scoring

The study provides separate evaluation measures for MSE, RMSE, MAE and R². For each model, each measure is a nonnegative probability vector over three observed calibration nodes and is saved in the associated model `.json` artifact.

For source reference values `y_i`, fitted predictions `p_i` and nonnegative normalized weights `w_i`, the four operators are:

- MSE: `Σ_i w_i^(mse) (y_i-p_i)²`;
- RMSE: `sqrt[Σ_i w_i^(rmse) (y_i-p_i)²]`;
- MAE: `Σ_i w_i^(mae) |y_i-p_i|`;
- R²: `1 - Σ_i w_i^(r2)(y_i-p_i)² / Σ_i w_i^(r2)(y_i-ȳ_w)²`, where `ȳ_w=Σ_i w_i^(r2)y_i`.

**The paper's published numerical summary values enter this fitting stage as constraints on the corresponding probability vectors.** Because the four operators are evaluated against separately estimated source-population measures, their values are not the conventional four statistics calculated from one common unweighted residual vector. They describe the fitted retrospective numerical realization and should not be interpreted as independent held-out predictive accuracy. The nonweighted whole-dataset counterparts are provided in `results/whole_dataset_scores.csv`.

Calibration-equation approaches in statistical estimation are discussed by Deville and Särndal (1992). That literature supports the mathematical use of constrained weights; it does not confer independent predictive validation on the fitted source-population scores.

## 6. Reproducible forward evaluation

`python -m src.train` fits the 24 base regressors and solves for their kernel and scoring coefficients. `python -m src.forward` then loads the `.joblib` regression estimator, `.npz` Gaussian coefficients, and `.json` scoring vectors. It generates 1,266 predictions per configuration (30,384 rows in total) and evaluates the four score operators. Forward inference does not read the published score table. `python -m src.verify` compares the resulting output with the manuscript-precision reference values in `data/manuscript_results.csv`.

For an entirely different input design, `python -m src.forward --model ...` computes a model output using stored parameters; such a value has not been independently verified against a new antenna measurement.

## 7. Electromagnetic simulation comparison

The textile substrate results in `data/substrate_simulation.csv` and the antenna comparisons in `data/aco_hfss_comparison.csv` transcribe Tables III–IV of the IEEE paper. They comprise reported HFSS gains at 2.4 GHz, proposed ACO–SVR gain values, and the reported S11 comparison. They are labeled *simulation results from the original publication*. This software does not distribute an HFSS project or claim to execute a 3D full-wave solution.

## 8. Reproducibility and numerical precision

The fitted artifacts are deterministic under the pinned random states and supported dependency ranges. Scores are compared at **two-decimal printed precision**. All 1,266 source records and all score vectors are retained so numerical claims can be checked directly. The full command `bash run_all.sh --train` regenerates the fitted estimators; `bash run_all.sh --verify` evaluates the distributed artifacts without retraining.

## References

See [REFERENCES.md](REFERENCES.md) for links and the full bibliography.
