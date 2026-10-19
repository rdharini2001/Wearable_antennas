"""Conditional response distributions when operating frequency is unobserved.

An antenna geometry identifies a response curve, but not the sampled frequency.
This estimator represents the response distribution using only training sweeps.
Reported paper scores are not used to fit distributions or intervals.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .discovery import ROOT, observations, geometry_groups, _split_indices, metrics, GEOMETRY, SEED


def crps_sample(y: float, samples: np.ndarray) -> float:
    """Continuous ranked probability score of a finite empirical distribution."""
    samples = np.asarray(samples, float)
    if not len(samples):
        raise ValueError('A forecast needs at least one training observation')
    first = np.mean(np.abs(samples - float(y)))
    ordered = np.sort(samples)
    # 1/(2n²) Σ_i,j |x_i-x_j| = Σ_i (2i-n-1) x_(i) / n².
    n = len(ordered)
    second = np.dot(2*np.arange(1,n+1)-n-1, ordered) / n**2
    return float(first - second)


def conformal_quantile(scores: np.ndarray, alpha=0.1) -> float:
    """Split-conformal residual cutoff at ceil((n+1)(1-alpha)) / n."""
    scores = np.sort(np.asarray(scores, float))
    if not len(scores):
        raise ValueError('Empty calibration set')
    k = int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(scores[k-1]) if k <= len(scores) else float('inf')


@dataclass
class MarginalResponse:
    """Geometry-conditioned empirical response forecast (GHz omitted)."""
    geometries: np.ndarray
    samples: list[np.ndarray]
    scaler: StandardScaler
    interval_expansion: float
    quantile_level: float = 0.9

    def _lookup(self, geometry):
        x = np.asarray(geometry,float).reshape(1,-1)
        if x.shape[1] != 4 or not np.isfinite(x).all():
            raise ValueError('Expected four finite antenna geometry parameters')
        equal = np.all(np.isclose(self.geometries, x, atol=1e-9, rtol=0),axis=1)
        if equal.any():
            return self.samples[int(np.where(equal)[0][0])]
        # Transfer to unseen geometries is reported separately. This mixture
        # is not used to score already observed geometry distributions.
        a = self.scaler.transform(self.geometries)
        b = self.scaler.transform(x)[0]
        distance = np.linalg.norm(a-b,axis=1)
        neighbors = np.argsort(distance)[:min(3,len(a))]
        weights = np.exp(-distance[neighbors]/max(np.median(distance[neighbors]),.1))
        weights /= weights.sum()
        # Quantile-based stratified discretization of a weighted mixture.
        q = (np.arange(256)+.5)/256
        parts = []
        for pos,w in zip(neighbors,weights):
            take=max(1,int(np.round(256*w)))
            parts.append(np.quantile(self.samples[int(pos)],q[:take] if take==256 else (np.arange(take)+.5)/take))
        return np.concatenate(parts)

    def samples_for(self, geometry):
        return self._lookup(geometry)

    def predict(self, X, statistic='mean'):
        X=np.atleast_2d(np.asarray(X,float))
        out=[]
        for x in X:
            a=self._lookup(x)
            out.append(float(np.mean(a) if statistic=='mean' else np.median(a)) if statistic in ('mean','median') else float('nan'))
        if statistic not in ('mean','median'):
            raise ValueError('Use mean or median')
        return np.asarray(out)

    def interval(self, X):
        X=np.atleast_2d(np.asarray(X,float))
        alpha=(1-self.quantile_level)/2
        values=np.asarray([np.quantile(self._lookup(x),[alpha,1-alpha]) for x in X])
        values[:,0]-=self.interval_expansion
        values[:,1]+=self.interval_expansion
        return values


def fit_marginal(frame, group, train, seed=SEED, nominal=0.9):
    # Calibration rows are disjoint from distribution-fitting rows.
    fit,cal = train_test_split(train,test_size=.2,random_state=seed+137,stratify=group[train])
    geometry=frame[list(GEOMETRY)].to_numpy(float)
    y=frame.Strength.to_numpy(float)
    gids=np.unique(group[fit]); gg=[]; samples=[]
    for g in gids:
        rows=fit[group[fit]==g]
        gg.append(geometry[rows[0]])
        samples.append(y[rows].copy())
    geo=np.asarray(gg)
    forecaster=MarginalResponse(geo,samples,StandardScaler().fit(geo),0.0,nominal)
    base_intervals=forecaster.interval(geometry[cal])
    nonconformity=np.maximum(base_intervals[:,0]-y[cal],y[cal]-base_intervals[:,1])
    forecaster.interval_expansion=max(0.,conformal_quantile(nonconformity,alpha=1-nominal))
    return forecaster, dict(training_distribution_rows=len(fit),calibration_rows=len(cal),
                            conditional_designs=len(geo),nominal_coverage=nominal,
                            interval_expansion_db=forecaster.interval_expansion)


def _score_oracle_floor(y, group):
    # Evaluation diagnostics only; explicitly not used to fit any model.
    means={g:np.mean(y[group==g]) for g in np.unique(group)}
    medians={g:np.median(y[group==g]) for g in np.unique(group)}
    by_mean=np.asarray([means[g] for g in group]);by_median=np.asarray([medians[g] for g in group])
    a=metrics(y,by_mean)
    return dict(mse_minimum=float(a['mse']),rmse_minimum=float(a['rmse']),
                r2_maximum=float(a['r2']),mae_minimum=float(np.mean(np.abs(y-by_median))),
                basis='Evaluation-partition oracle constants; not available during training')


def stability(frame, groups, seeds=(41, 101, 2023, 3907)):
    """Independent held-out frequency subsets and calibration partitions."""
    x=frame[list(GEOMETRY)].to_numpy(float)
    y=frame.Strength.to_numpy(float)
    outcomes=[]
    for seed in seeds:
        train,test=_split_indices(frame,groups,'frequency_interpolation',seed)
        model,_=fit_marginal(frame,groups,train,seed)
        yy=y[test];xx=x[test]
        intervals=model.interval(xx)
        empirical_crps=np.mean([crps_sample(label,model.samples_for(g)) for label,g in zip(yy,xx)])
        outcomes.append(dict(seed=seed,n_test=len(test),crps_db=float(empirical_crps),
                             coverage_90=float(np.mean((yy>=intervals[:,0])&(yy<=intervals[:,1]))),
                             mean_width_db=float(np.mean(intervals[:,1]-intervals[:,0])),
                             median_point_mae_db=float(np.mean(np.abs(yy-model.predict(xx,'median'))))))
    return pd.DataFrame(outcomes)


def run(data=None, output=None, seed=SEED):
    frame=observations(data);groups=geometry_groups(frame)
    train,test=_split_indices(frame,groups,'frequency_interpolation',seed)
    x=frame[list(GEOMETRY)].to_numpy(float)
    y=frame.Strength.to_numpy(float)
    predictor,recipe=fit_marginal(frame,groups,train,seed)
    test_y=y[test];test_x=x[test];test_group=groups[test]
    point_mean=predictor.predict(test_x,'mean')
    point_med=predictor.predict(test_x,'median')
    intervals=predictor.interval(test_x)
    crps=np.asarray([crps_sample(label,predictor.samples_for(g)) for label,g in zip(test_y,test_x)])
    coverage=(test_y>=intervals[:,0])&(test_y<=intervals[:,1])
    out=Path(output or ROOT/'results'/'marginal');out.mkdir(parents=True,exist_ok=True)
    joblib.dump(predictor,out/'frequency_blind_distribution.joblib',compress=3)
    scores={'pointwise_mean':metrics(test_y,point_mean),
            'pointwise_median':metrics(test_y,point_med),
            'probabilistic':dict(crps_db=float(crps.mean()),interval_coverage=float(coverage.mean()),
                                 mean_interval_width_db=float(np.mean(intervals[:,1]-intervals[:,0])),
                                 nominal_coverage=recipe['nominal_coverage']),
            'test_oracle_floor':_score_oracle_floor(test_y,test_group),
            'calibration':recipe,'test_rows':len(test),'seed':seed}
    scores['probabilistic']['crps_improvement_over_median_point_db']=float(np.mean(np.abs(test_y-point_med))-np.mean(crps))
    pd.DataFrame(dict(row=test,group=test_group,observed_db=test_y,mean_db=point_mean,
                      median_db=point_med,interval_lower_db=intervals[:,0],
                      interval_upper_db=intervals[:,1],crps_db=crps,covered=coverage)).to_csv(out/'forecasts.csv',index=False)
    # Test transfer to previously unseen antenna geometries separately.
    train_new,test_new=_split_indices(frame,groups,'geometry_holdout',seed)
    transfer,transfer_spec=fit_marginal(frame,groups,train_new,seed)
    x_new=x[test_new];y_new=y[test_new]
    mu_new=transfer.predict(x_new)
    bounds_new=transfer.interval(x_new)
    scores['geometry_transfer']={
        'pointwise':metrics(y_new,mu_new),
        'probabilistic':{
            'crps_db':float(np.mean([crps_sample(v,transfer.samples_for(z)) for v,z in zip(y_new,x_new)])),
            'interval_coverage':float(np.mean((y_new>=bounds_new[:,0])&(y_new<=bounds_new[:,1]))),
            'mean_interval_width_db':float(np.mean(bounds_new[:,1]-bounds_new[:,0]))
        },
        'n_test_rows':int(len(test_new)),
        'n_unseen_geometries':int(len(np.unique(groups[test_new]))),
        'calibration':transfer_spec
    }
    robustness=stability(frame,groups)
    robustness.to_csv(out/'split_stability.csv',index=False)
    scores['robustness']={
        'seeds':robustness.seed.tolist(),
        'crps_range_db':[float(robustness.crps_db.min()),float(robustness.crps_db.max())],
        'interval_coverage_range':[float(robustness.coverage_90.min()),float(robustness.coverage_90.max())]
    }
    (out/'metrics.json').write_text(json.dumps(scores,indent=2)+'\n')
    print('Frequency-blind probabilistic evaluation: '+json.dumps(scores['probabilistic']))
    return scores

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data');parser.add_argument('--output');parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    from . import marginal as module
    module.run(args.data,args.output,args.seed)
