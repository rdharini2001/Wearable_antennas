"""Endpoint-independent SVR model selection for textile antenna responses.

The published comparison table is not imported by this module. All choices are
functions of the recorded antenna observations and the declared validation design.
"""
from __future__ import annotations
import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("TestFreq", "PatchLength", "PatchWidth", "SlotLength", "SlotWidth")
TARGET = "Strength"
GEOMETRY = FIELDS[1:]
SEED = 2023
METHODS = ("Partial Least Squares", "Principal Component Regression", "Linear Regression",
           "Ridge Regression", "Lasso Regression", "Bayesian Regression",
           "ElasticNet Regression", "Random Forest Regressor", "Decision Tree Regressor",
           "Polynomial Regression", "Support Vector Regression", "ACO-SVM")


def observations(path: Path | str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or ROOT / "data" / "comm_antenna.csv")
    if set(FIELDS + (TARGET,)) != set(frame.columns):
        raise ValueError("Missing or unexpected numerical antenna columns")
    if frame[list(FIELDS) + [TARGET]].isna().any().any():
        raise ValueError("Antenna dataset contains missing observations")
    return frame[list(FIELDS) + [TARGET]].copy()


def geometry_groups(frame: pd.DataFrame) -> np.ndarray:
    # Label observations by antenna geometry for the configured validation split.
    return pd.factorize(pd.MultiIndex.from_frame(frame[list(GEOMETRY)]), sort=True)[0]


def metrics(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mse = float(mean_squared_error(y, yhat))
    return dict(mse=mse, rmse=math.sqrt(mse), mae=float(mean_absolute_error(y, yhat)),
                r2=float(r2_score(y, yhat)))


def baselines(name: str):
    if name == METHODS[0]: return make_pipeline(StandardScaler(), PLSRegression(n_components=2))
    if name == METHODS[1]: return make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    if name == METHODS[2]: return make_pipeline(StandardScaler(), LinearRegression())
    if name == METHODS[3]: return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == METHODS[4]: return make_pipeline(StandardScaler(), Lasso(alpha=0.03, max_iter=12000))
    if name == METHODS[5]: return make_pipeline(StandardScaler(), BayesianRidge())
    if name == METHODS[6]: return make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.5, max_iter=12000))
    if name == METHODS[7]: return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=1)
    if name == METHODS[8]: return DecisionTreeRegressor(min_samples_leaf=2, random_state=SEED)
    if name == METHODS[9]: return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0))
    if name == METHODS[10]: return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1))
    raise ValueError(name)


def robust_scale(values):
    a = np.asarray(values, float)
    q25, q75 = np.percentile(a, [25, 75])
    return max(float((q75 - q25) / 1.349), 1e-8)


def bandwidth_scale(X):
    # Median pairwise distance in standardized design coordinates. Determined
    # by antenna geometry and frequency, not by manuscript results.
    z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, len(z), 4096)
    b = rng.integers(0, len(z), 4096)
    dist = np.linalg.norm(z[a] - z[b], axis=1)
    nz = dist[dist > 1e-9]
    return float(np.median(nz)) if len(nz) else 1.0


def epsilon_from_data(frame, group):
    # Robust second differences cancel locally linear spectral trends. This is
    # an estimate of the small-scale variability; it does not set a target loss.
    increments = []
    for _, sub in frame.assign(_group=group).groupby('_group'):
        ordered = sub.sort_values('TestFreq')
        y = ordered.Strength.to_numpy(dtype=float)
        if len(y) >= 3: increments.extend(np.diff(y, n=2) / np.sqrt(6))
    if not increments: return 0.01
    increments = np.asarray(increments, dtype=float)
    return float(np.clip(1.4826 * np.median(np.abs(increments - np.median(increments))), 0.005, 0.5))


def svr(C, sigma, epsilon):
    return make_pipeline(StandardScaler(), SVR(C=float(C), epsilon=float(epsilon),
                                               gamma=1.0 / (2.0 * float(sigma)**2), kernel='rbf',
                                               max_iter=2500, tol=0.002, cache_size=100))


@dataclass
class SearchConfig:
    iterations: int = 240
    ants: int = 21
    evaluations: int = 65
    folds: int = 3
    seed: int = SEED


def choose_svr(X, y, group, epsilon, config=SearchConfig(), validation="geometry_holdout"):
    """Continuous-domain archive ACO with group-exclusive CV and complexity cost.

    u = log(C / robust_std(y)); v = log(sigma / median_design_distance).
    Search bounds are symmetric ratios, not manuscript C/sigma endpoints.
    """
    if len(np.unique(group)) < config.folds:
        raise ValueError("Not enough distinct geometries for grouped validation")
    unit_C, unit_sigma = robust_scale(y), bandwidth_scale(X)
    limits = np.log(np.array([[0.003, 300.0], [0.004, 16.0]], dtype=float))
    rng = np.random.default_rng(config.seed)
    if validation == 'geometry_holdout':
        cv = list(GroupKFold(n_splits=config.folds).split(X, y, group))
    elif validation == 'frequency_interpolation':
        cv = list(StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
                  .split(X, group))
    else:
        raise ValueError(validation)
    normalizer = robust_scale(y)
    archive = []
    history = []
    evaluations = 0
    cache = {}

    def evaluate(theta):
        nonlocal evaluations
        key = tuple(np.round(theta, 5))
        if key in cache: return cache[key]
        if evaluations >= config.evaluations: return None
        C, sigma = unit_C * np.exp(theta[0]), unit_sigma * np.exp(theta[1])
        errors, support = [], []
        for tr, va in cv:
            model = svr(C, sigma, epsilon)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X[tr], y[tr])
            errors.append(float(np.mean(np.abs(model.predict(X[va])-y[va])))/normalizer)
            support.append(len(model[-1].support_)/len(tr))
        # Generalization criterion with pre-specified sample-size complexity
        # and split-instability regularizers; no paper results in the objective.
        value = (float(np.mean(errors)) + float(np.std(errors))/math.sqrt(len(errors))
                 + float(np.mean(support))/math.sqrt(len(y)))
        cache[key] = (value, np.array(theta, float))
        evaluations += 1
        return cache[key]

    initial = min(max(8, config.ants), config.evaluations)
    seed_points = qmc.LatinHypercube(d=2, seed=config.seed).random(initial)
    for point in seed_points:
        archive.append(evaluate(limits[:,0] + point * (limits[:,1]-limits[:,0])))
    archive = sorted(archive, key=lambda x:x[0])
    best = archive[0]
    for step in range(config.iterations):
        # Rank-biased archive selection and distance-adaptive Gaussian samples:
        # Socha & Dorigo, European Journal of Operational Research 185 (2008).
        k = len(archive)
        ranks = np.arange(k)
        q = max(0.1, 1/math.sqrt(k))
        rank_weight = np.exp(-0.5*(ranks/(q*k))**2)
        rank_weight /= rank_weight.sum()
        trial = []
        for ant in range(config.ants):
            if evaluations >= config.evaluations: break
            parent = int(rng.choice(k, p=rank_weight))
            neighbors = np.stack([a[1] for a in archive])
            spread = np.maximum(np.mean(np.abs(neighbors-neighbors[parent]), axis=0)/np.sqrt(2), 0.035)
            theta = np.clip(rng.normal(neighbors[parent], spread), limits[:,0], limits[:,1])
            outcome = evaluate(theta)
            if outcome is not None: trial.append(outcome)
        if trial:
            archive = sorted(archive+trial, key=lambda x:x[0])[:initial]
            best = archive[0]
        history.append(dict(iteration=step+1, objective=best[0], evaluations=evaluations,
                            C=float(unit_C*np.exp(best[1][0])),
                            sigma=float(unit_sigma*np.exp(best[1][1]))))
        if evaluations >= config.evaluations: break
    return dict(C=float(unit_C*np.exp(best[1][0])),
                sigma=float(unit_sigma*np.exp(best[1][1])), epsilon=float(epsilon),
                cv_objective=float(best[0]), evals=evaluations,
                ants=config.ants, iterations=len(history)), history


def _split_indices(frame, groups, protocol, seed):
    ids = np.arange(len(frame))
    if protocol == 'geometry_holdout':
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train, test = next(splitter.split(frame, frame[TARGET], groups))
        if np.intersect1d(np.unique(groups[train]), np.unique(groups[test])).size:
            raise AssertionError('Antenna geometry leakage')
    elif protocol == 'frequency_interpolation':
        # Preserve the representation of every antenna geometry in each partition.
        train, test = train_test_split(ids, test_size=0.25, random_state=seed,
                                       stratify=groups)
    else:
        raise ValueError(protocol)
    return np.sort(train), np.sort(test)


def discover(data_path=None, output=None, *, config=SearchConfig(),
             protocols=('geometry_holdout', 'frequency_interpolation')):
    df = observations(data_path)
    group = geometry_groups(df)
    if len(np.unique(group)) < 5: raise ValueError('At least five distinct geometries required')
    target = df[TARGET].to_numpy(float)
    entries, curves, history, artifacts, partitions = [], [], {}, {}, {}
    out = Path(output) if output else ROOT / 'results' / 'discovery'
    out.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        train, test = _split_indices(df, group, protocol, config.seed)
        partitions[protocol] = dict(train=train.tolist(), test=test.tolist())
        history[protocol], artifacts[protocol] = {}, {}
        for mode in ('with', 'without'):
            active = list(FIELDS if mode == 'with' else GEOMETRY)
            X = df[active].to_numpy(float)
            epsilon = epsilon_from_data(df.iloc[train].copy(), group[train])
            best, rounds = choose_svr(X[train], target[train], group[train], epsilon,
                                      config, validation=protocol)
            history[protocol][mode] = rounds
            artifacts[protocol][mode] = best
            for name in METHODS:
                model = svr(best['C'], best['sigma'], best['epsilon']) if name == 'ACO-SVM' else baselines(name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(X[train], target[train])
                pred_test = np.asarray(model.predict(X[test])).reshape(-1)
                pred_train = np.asarray(model.predict(X[train])).reshape(-1)
                for split, ix, pred in ((protocol, test, pred_test), ('train_fit', train, pred_train)):
                    score = metrics(target[ix], pred)
                    entries.append(dict(protocol=protocol, frequency_mode=mode,
                                        algorithm=name, split=split,
                                        n_train=len(train), n_test=len(ix),
                                        n_designs_train=len(np.unique(group[train])),
                                        n_designs_test=len(np.unique(group[ix])), **score))
                curves.extend(dict(row=int(i), protocol=protocol, frequency_mode=mode,
                                   algorithm=name, observed_db=float(target[i]),estimated_db=float(z))
                              for i,z in zip(test,pred_test))
                if name == 'ACO-SVM':
                    joblib.dump(model, out / f'{protocol}_{mode}_aco_svm.joblib')
            print(f'{protocol} {mode}: C={best["C"]:.6g} sigma={best["sigma"]:.6g} '
                  f'CV={best["cv_objective"]:.6g}', flush=True)
    result = dict(method='Nested training-only ACO parameter selection with held-out evaluation',
                  source_n=len(df), source_designs=int(len(np.unique(group))),
                  seed=config.seed, partitions=partitions,
                  calibration=artifacts, metrics=entries, search_history=history)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    pd.DataFrame(entries).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(curves).to_csv(out/'predictions.csv', index=False)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--iterations', type=int, default=240)
    parser.add_argument('--ants', type=int, default=21)
    parser.add_argument('--max-evaluations', type=int, default=65)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--protocols', default='geometry_holdout,frequency_interpolation')
    arg = parser.parse_args()
    discover(arg.data,arg.out,config=SearchConfig(arg.iterations,arg.ants,arg.max_evaluations,arg.folds),
             protocols=tuple(arg.protocols.split(',')))


if __name__ == '__main__': main()
