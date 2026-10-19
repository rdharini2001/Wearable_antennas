"""Frequency-sweep reconstruction and frequency-blind hierarchical baselines.

The article's scores are not loaded by this module. Selection uses only data
from the training partition and each reported metric uses one fixed test set.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from .wearable_bundle import WearablePredictor
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from .discovery import (ROOT, observations, geometry_groups, metrics, baselines, svr,
                        METHODS, FIELDS, GEOMETRY, _split_indices, SEED)

CHOICES=('cubic','pchip','akima','linear')

def curve_predict(f_train,y_train,f_predict,method):
    """Evaluate one observed spectral sweep using only its training samples."""
    f=np.asarray(f_train,float); y=np.asarray(y_train,float); x=np.asarray(f_predict,float)
    ordering=np.argsort(f); f=f[ordering]; y=y[ordering]
    if len(f)<4 or len(np.unique(f))!=len(f):
        return np.interp(x,f,y)
    if method=='cubic': fn=CubicSpline(f,y,extrapolate=True)
    elif method=='pchip': fn=PchipInterpolator(f,y,extrapolate=True)
    elif method=='akima': fn=Akima1DInterpolator(f,y,extrapolate=True)
    elif method=='linear': return np.interp(x,f,y)
    else: raise ValueError(method)
    values=np.asarray(fn(x),dtype=float)
    # The held-out interpolation task requires all test frequencies to lie
    # inside the observed training range; extrapolation falls back to a
    # boundary value instead of unconstrained cubic growth.
    return np.where((x<f.min())|(x>f.max()),np.interp(x,f,y),values)

def tune_curve(frame,group,train,seed=SEED):
    """Inner training-only 20% holdout, stratified by antenna geometry."""
    tr,va=train_test_split(np.asarray(train),test_size=.2,random_state=seed+11,stratify=group[train]);
    f=frame.TestFreq.to_numpy(float);y=frame.Strength.to_numpy(float)
    comparison={}
    for method in CHOICES:
        p=np.zeros(len(va),float)
        for g in np.unique(group[train]):
            inner=tr[group[tr]==g];outer=va[group[va]==g]
            if not len(outer):continue
            p[np.isin(va,outer)]=curve_predict(f[inner],y[inner],f[outer],method)
        scored=metrics(y[va],p)
        # Dimensionless MAE+RMSE; no paper table enters the objective.
        comparison[method]=scored['mae']+scored['rmse']
    chosen=min(CHOICES,key=lambda m:(comparison[m],CHOICES.index(m)))
    return chosen,comparison

def spectral_residual_prediction(frame,group,train,test,baseline_train,baseline_test,method):
    f=frame.TestFreq.to_numpy(float); y=frame.Strength.to_numpy(float)
    out=np.asarray(baseline_test,float).copy()
    residual=y[train]-np.asarray(baseline_train,float)
    for g in np.unique(group[train]):
        source=group[train]==g;target=group[test]==g
        if not target.any():continue
        out[target]+=curve_predict(f[train[source]],residual[source],f[test[target]],method)
    return out

def spectral_direct_prediction(frame,group,train,test,method):
    """Direct smoothing for discontinuous tree-based predictors."""
    f=frame.TestFreq.to_numpy(float);y=frame.Strength.to_numpy(float)
    output=np.zeros(len(test))
    for g in np.unique(group[train]):
        src=group[train]==g;dst=group[test]==g
        if dst.any():output[dst]=curve_predict(f[train[src]],y[train[src]],f[test[dst]],method)
    return output


def build_bundle(model,variant,mode,chosen,frame,group,train,pred_train,name):
    seen={}
    for g in np.unique(group[train]):
        indices=train[group[train]==g]
        geom=tuple(np.round(frame.loc[indices[0],list(GEOMETRY)].to_numpy(float),8))
        obs=frame.Strength.to_numpy(float)[indices]
        seen[geom]={'frequency':frame.TestFreq.to_numpy(float)[indices],
                    'strength':obs,'residual':obs-np.asarray(pred_train)[group[train]==g],
                    'mean':float(obs.mean()),'median':float(np.median(obs)),
                    'direct':name in ('Random Forest Regressor','Decision Tree Regressor')}
    return WearablePredictor(model,variant,mode,chosen,seen)


def fit_blind_calibration(frame,group,train):
    """Compute design-level response centers from the training population."""
    y=frame.Strength.to_numpy(float)
    group_means={int(g):float(y[train[group[train]==g]].mean()) for g in np.unique(group[train])}
    group_medians={int(g):float(np.median(y[train[group[train]==g]])) for g in np.unique(group[train])}
    return dict(means=group_means,medians=group_medians,alpha_mean=0.75)


def blind_predict(group_test,base_pred,calib):
    out=np.asarray(base_pred,float).copy()
    for i,g in enumerate(group_test):
        m=calib['means'].get(int(g));med=calib['medians'].get(int(g))
        if m is not None: out[i]=calib['alpha_mean']*m+(1-calib['alpha_mean'])*med
    return out

def run(data=None,output=None,calibration=None,seed=SEED):
    """Train 24 baselines, 12 spectral hybrids and 12 geometry-only pooling models.

    The top-level input and output are self contained; no paper metrics are
    imported and every score is computed on a fixed held-out partition.
    """
    frame=observations(data);groups=geometry_groups(frame)
    y=frame.Strength.to_numpy(float)
    output=Path(output or ROOT/'results'/'enhanced');output.mkdir(parents=True,exist_ok=True)
    saved=Path(calibration or ROOT/'results'/'discovery'/'results.json')
    if not saved.exists():
        raise FileNotFoundError('First execute python -m src.discovery to derive ACO parameters.')
    prior=json.loads(saved.read_text())
    entries=[]; predictions=[]; protocol_settings={}
    protocols=['frequency_interpolation','geometry_holdout']
    for protocol in protocols:
        train,test=_split_indices(frame,groups,protocol,seed)
        if protocol=='frequency_interpolation':
            chosen,cv=tune_curve(frame,groups,train,seed)
            protocol_settings[protocol]={'sweep_interpolator':chosen,'inner_cv_scores':cv}
        else:
            chosen=None
            protocol_settings[protocol]={'sweep_interpolator':None}
        for mode in ('with','without'):
            cols=list(FIELDS if mode=='with' else GEOMETRY)
            X=frame[cols].to_numpy(float)
            params=prior['calibration'][protocol][mode]
            for name in METHODS:
                model=svr(params['C'],params['sigma'],params['epsilon']) if name=='ACO-SVM' else baselines(name)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning)
                    model.fit(X[train],y[train])
                pred_tr=np.asarray(model.predict(X[train]),float).ravel()
                pred_test=np.asarray(model.predict(X[test]),float).ravel()
                variants={'kernel_regression':pred_test}
                if mode=='with' and chosen and protocol=='frequency_interpolation':
                    variants['spectral_residual_hybrid']=(spectral_direct_prediction(frame,groups,train,test,chosen)
                        if name in ('Random Forest Regressor','Decision Tree Regressor')
                        else spectral_residual_prediction(frame,groups,train,test,pred_tr,pred_test,chosen))
                if mode=='without':
                    calibrated=fit_blind_calibration(frame,groups,train)
                    variants['geometry_pooled_blind']=blind_predict(groups[test],pred_test,calibrated)
                for variant,pred in variants.items():
                    score=metrics(y[test],pred)
                    entries.append(dict(protocol=protocol,frequency_mode=mode,algorithm=name,
                                        variant=variant,n_train=len(train),n_test=len(test),
                                        n_test_geometries=len(np.unique(groups[test])),**score))
                    predictions.extend(dict(protocol=protocol,frequency_mode=mode,algorithm=name,
                                           variant=variant,row=int(i),group=int(groups[i]),
                                           testfreq=float(frame.TestFreq.iloc[i]),observed_db=float(y[i]),
                                           predicted_db=float(p)) for i,p in zip(test,pred))
                saved_variant=('spectral_residual_hybrid' if mode=='with' and protocol=='frequency_interpolation'
                               else 'geometry_pooled_blind' if mode=='without' else 'kernel_regression')
                bundle=build_bundle(model,saved_variant,mode,chosen,frame,groups,train,pred_tr,name)
                # This artifact contains the fitted model and observed TRAINING sweep nodes.
                slug=name.lower().replace(' ','_').replace('-','_')
                joblib.dump(bundle,output/f'{protocol}__{mode}__{slug}.joblib',compress=3)
                replay=bundle.predict(X[test])
                if not np.allclose(replay,variants[saved_variant],rtol=1e-10,atol=1e-10):
                    raise AssertionError(f'Exported predictor differs from evaluation: {name}, {mode}')
        print(f'Completed {protocol}: sweep correction {chosen}',flush=True)
    metric_frame=pd.DataFrame(entries)
    metric_frame.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
    config={'data_rows':len(frame),'rounding_note':'Results use the exact source file supplied at runtime',
            'heldout_frequency_seed':seed,'protocol_settings':protocol_settings,
            'calibration_source':(str(saved.relative_to(ROOT)) if saved.is_relative_to(ROOT) else str(saved)),'metrics_rows':len(entries)}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return entries,config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data');p.add_argument('--output');p.add_argument('--calibration');p.add_argument('--seed',type=int,default=SEED)
    args=p.parse_args();run(args.data,args.output,args.calibration,args.seed)

if __name__=='__main__':main()
