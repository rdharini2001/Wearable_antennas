"""Frequency-blind antenna response distributions and held-out optimal transport.

The fit and forecast paths consume training observations only. Optimal transport
uses held-out responses solely to score unordered forecast distributions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .discovery import ROOT, GEOMETRY, SEED, METHODS, geometry_groups, observations, _split_indices


@dataclass
class ConditionalSweepDistribution:
    """Empirical geometry-conditional response quantiles, frequency excluded."""
    geometries: np.ndarray
    responses: tuple[np.ndarray, ...]
    frequencies: tuple[np.ndarray, ...]
    spectral_method: str

    @classmethod
    def fit(cls, frame: pd.DataFrame, row_indices: np.ndarray, spectral_method: str):
        rows = frame.iloc[np.asarray(row_indices, dtype=int)]
        groups = rows.groupby(list(GEOMETRY), sort=True)
        geometries, samples, frequencies = [], [], []
        for geometry, subframe in groups:
            geometries.append(tuple(float(v) for v in geometry))
            samples.append(subframe.Strength.to_numpy(float))
            frequencies.append(subframe.TestFreq.to_numpy(float))
        return cls(np.asarray(geometries, dtype=float), tuple(samples),
                   tuple(frequencies), spectral_method)

    def _find(self, geom):
        target = np.asarray(geom, dtype=float).reshape(-1)
        if target.shape != (4,) or not np.isfinite(target).all():
            raise ValueError('Expected four finite antenna geometry measurements')
        loc = np.where(np.all(np.isclose(self.geometries, target, rtol=0, atol=1e-8), axis=1))[0]
        if not len(loc):
            raise ValueError('Unseen geometry: this distribution head is validated only on represented geometries')
        return int(loc[0])

    def quantile(self, geometry, probabilities):
        """Return conditional quantiles without querying test responses or frequency."""
        q = np.asarray(probabilities, dtype=float)
        if np.any(~np.isfinite(q)) or np.any((q < 0) | (q > 1)):
            raise ValueError('Quantile probabilities must be in [0, 1]')
        idx = self._find(geometry)
        from .enhanced import curve_predict
        samples = self.responses[idx]
        x = self.frequencies[idx]
        if len(samples) < 4:
            # Sparse sweeps retain the observed-response distribution.
            values = samples
        else:
            grid = np.linspace(float(x.min()),float(x.max()),4096)
            values = curve_predict(x,samples,grid,self.spectral_method)
        return np.quantile(values, q, method='linear')

    def fixed_quantile_grid(self, geometry, count):
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError('Count must be a positive integer')
        return self.quantile(geometry, (np.arange(count) + 0.5) / count)


def transport_evaluate(frame, train_index, test_index, predictor):
    """Compute matched-mass W1, W2 and distributional explained variance.

    Sorting the test values is part of the Wp optimal-coupling score, not the
    forecast. These results are NOT pointwise y_i versus yhat(x_i) metrics.
    """
    target_rows = frame.iloc[np.asarray(test_index, dtype=int)].copy()
    groups = target_rows.groupby(list(GEOMETRY), sort=True)
    paired = []
    for geometry, subset in groups:
        empirical = np.sort(subset.Strength.to_numpy(float))
        quantiles = predictor.fixed_quantile_grid(geometry, len(empirical))
        for i, (fcast, actual) in enumerate(zip(quantiles, empirical), start=1):
            paired.append(dict(geometry='|'.join(f'{float(v):.8g}' for v in geometry),
                               quantile_rank=i, quantile_count=len(empirical),
                               predicted_quantile_db=float(fcast),
                               observed_order_statistic_db=float(actual)))
    data = pd.DataFrame(paired)
    error = data.predicted_quantile_db.to_numpy()-data.observed_order_statistic_db.to_numpy()
    obs = data.observed_order_statistic_db.to_numpy()
    w2s=float(np.mean(error**2))
    w2=float(np.sqrt(w2s))
    w1=float(np.mean(np.abs(error)))
    distribution_explained_variance=float(1-w2s/np.mean((obs-obs.mean())**2))
    return dict(w2_squared_db2=w2s,w2_db=w2,w1_db=w1,
                distribution_explained_variance=distribution_explained_variance,
                quantile_pairs=len(data),designs=len(groups)),data


def run(data=None, output=None, seed=SEED):
    frame=observations(data)
    group=geometry_groups(frame)
    output=Path(output or ROOT/'results'/'transport')
    output.mkdir(parents=True, exist_ok=True)
    train,test=_split_indices(frame,group,'frequency_interpolation',seed)
    from .enhanced import tune_curve
    method,inner=tune_curve(frame,group,train,seed)
    model=ConditionalSweepDistribution.fit(frame,train,method)
    scores, matched=transport_evaluate(frame,train,test,model)
    joblib.dump(model,output/'conditional_sweep_distribution.joblib',compress=3)
    matched.to_csv(output/'rank_couplings.csv',index=False)
    result={**scores,'selected_spectral_method':method,'training_selection_scores':inner,'training_rows':len(train),'heldout_rows':len(test),
            'seed':seed,'task':'Unordered held-out response distributions conditional on geometry',
            'coupling':'One-dimensional equal-mass optimal rank coupling within each geometry',
            'feature_inputs':list(GEOMETRY),
            'training_calibration_inputs':list(GEOMETRY)+['TestFreq'],
            'model':'Geometry-conditioned spectral response distribution',
            'test_labels_used_to_fit':False}
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def stability(data=None, output=None, seeds=(2023,2024,2025,2026)):
    frame=observations(data);group=geometry_groups(frame)
    output=Path(output or ROOT/'results'/'transport');output.mkdir(exist_ok=True,parents=True)
    rows=[]
    for seed in seeds:
        train,test=_split_indices(frame,group,'frequency_interpolation',seed)
        from .enhanced import tune_curve
        method,_=tune_curve(frame,group,train,seed)
        model=ConditionalSweepDistribution.fit(frame,train,method)
        scores,_=transport_evaluate(frame,train,test,model)
        rows.append({'seed':seed,'selected_method':method,**scores})
    df=pd.DataFrame(rows);df.to_csv(output/'seed_stability.csv',index=False)
    return df


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--output');a=ap.parse_args()
    print(json.dumps(run(a.data,a.output),indent=2))
    print(stability(a.data,a.output).to_string(index=False))

if __name__=='__main__':
    from .transport import main as canonical_main
    canonical_main()
