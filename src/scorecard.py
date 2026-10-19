"""Paper-reference comparison and domain-specific evaluation reports.

The reference file is used only after all model fitting and prediction finishes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .discovery import ROOT
from .audit import compatible_mse_rmse

METRICS=('mse','rmse','mae','r2')

def better_or_equal(value,paper,metric):
    return bool(value>=paper-1e-12 if metric=='r2' else value<=paper+1e-12)

def report(root=None,paper_file=None):
    root=Path(root or ROOT/'results'/'enhanced')
    paper=pd.read_csv(paper_file or ROOT/'data'/'manuscript_results.csv')
    measured=pd.read_csv(root/'metrics.csv')
    predictions=pd.read_csv(root/'predictions.csv')
    if paper.shape[0]!=24 or len(paper[['frequency_mode','algorithm']].drop_duplicates())!=24:
        raise ValueError('Expected 24 distinct manuscript reference configurations')
    if len(predictions.groupby(['protocol','frequency_mode','algorithm','variant']))!=len(measured):
        raise ValueError('Prediction and score partitions disagree')
    rows=[]
    for _,base in measured.iterrows():
        ref=paper[(paper.frequency_mode==base.frequency_mode)&(paper.algorithm==base.algorithm)]
        if len(ref)!=1:raise ValueError('Paper entry not found')
        ref=ref.iloc[0]
        for metric in METRICS:
            measured_value=float(base[metric]);reference_value=float(ref[metric]);
            rows.append(dict(protocol=base.protocol,frequency_mode=base.frequency_mode,
                    algorithm=base.algorithm,variant=base.variant,metric=metric,
                    paper=reference_value,observed=measured_value,
                    paper_or_better=better_or_equal(measured_value,reference_value,metric),
                    exact_two_decimal_match=bool(round(measured_value,2)==round(reference_value,2)),
                    same_pointwise_task=True,
                    note='Different held-out sampling from original paper; reference protocol not established'))
    # Antenna-level evaluation targets only the mean response per design. This
    # is appropriate for selecting frequency-agnostic designs, but its metrics
    # are not pointwise antenna-signal-strength error metrics.
    summaries=[];macro=[]
    for (protocol,mode,algorithm,variant),pred in predictions.groupby(['protocol','frequency_mode','algorithm','variant'],sort=False):
        aggregates=pred.groupby('group',as_index=False)[['observed_db','predicted_db']].mean()
        if len(aggregates)<2:continue
        y=aggregates.observed_db.to_numpy();yh=aggregates.predicted_db.to_numpy()
        mse=float(mean_squared_error(y,yh));rmse=float(np.sqrt(mse));mae=float(mean_absolute_error(y,yh))
        r2=float(r2_score(y,yh))
        summaries.append(dict(protocol=protocol,frequency_mode=mode,algorithm=algorithm,
                              variant=variant,n_geometries=len(y),mse=mse,rmse=rmse,mae=mae,r2=r2))
        if mode=='without':
            ref=paper[(paper.frequency_mode==mode)&(paper.algorithm==algorithm)].iloc[0]
            for metric,value in [('mse',mse),('rmse',rmse),('mae',mae),('r2',r2)]:
                macro.append(dict(protocol=protocol,frequency_mode=mode,algorithm=algorithm,
                                  variant=variant,metric=metric,paper=float(ref[metric]),observed=value,
                                  paper_or_better=better_or_equal(value,float(ref[metric]),metric),
                                  exact_two_decimal_match=round(value,2)==round(float(ref[metric]),2),
                                  same_pointwise_task=False,
                                  note='Design-mean prediction; numeric thresholds not directly comparable to pointwise paper metrics'))
    # Build a preferred representation of the prior report: spectral hybrid
    # with frequency, frequency-blind group pooling when frequency is absent.
    results=pd.DataFrame(rows)
    preferred=results[(results.protocol=='frequency_interpolation')&(
         ((results.frequency_mode=='with')&(results.variant=='spectral_residual_hybrid'))|
         ((results.frequency_mode=='without')&(results.variant=='geometry_pooled_blind')))].copy()
    if len(preferred)!=96:raise AssertionError('Must have exactly 96 preferred pointwise comparisons')
    def count(df):return {'met_or_better':int(df.paper_or_better.sum()),'exact':int(df.exact_two_decimal_match.sum()),'total':len(df)}
    grouped={m:count(x) for m,x in preferred.groupby('frequency_mode')}
    summary={'primary_protocol':'frequency_interpolation',
             'pointwise_thresholds':count(preferred), 'pointwise_by_frequency':grouped,
             'geometry_mean_thresholds_not_comparable':count(pd.DataFrame(macro)[
                 (pd.DataFrame(macro).protocol=='frequency_interpolation')&
                 (pd.DataFrame(macro).variant=='geometry_pooled_blind')]),
             'paper_mse_rmse_inconsistent_pairs':int(sum(not compatible_mse_rmse(row.mse,row.rmse) for row in paper.itertuples())),
             'interpretation':'Meeting a reference numeric threshold on a different holdout is not exact replication. Geometry means evaluate a distinct design-level target.'}
    results.to_csv(root/'all_pointwise_comparisons.csv',index=False)
    preferred.to_csv(root/'preferred_96_pointwise.csv',index=False)
    pd.DataFrame(summaries).to_csv(root/'geometry_means.csv',index=False)
    pd.DataFrame(macro).to_csv(root/'geometry_mean_thresholds_not_comparable.csv',index=False)
    # For this fixed test set, a geometry-only point estimator has oracle
    # metric extrema. These are post-evaluation feasibility diagnostics only.
    diagnostic=root.parent/'marginal'/'metrics.json'
    if diagnostic.exists():
        oracle=json.loads(diagnostic.read_text())['test_oracle_floor']
        extrema={'mse':oracle['mse_minimum'],'rmse':oracle['rmse_minimum'],
                 'mae':oracle['mae_minimum'],'r2':oracle['r2_maximum']}
        blocked=[]
        for ref in paper[paper.frequency_mode=='without'].itertuples():
            for metric,limit in extrema.items():
                if not better_or_equal(limit,float(getattr(ref,metric)),metric):
                    blocked.append(dict(algorithm=ref.algorithm,metric=metric,
                                        paper=float(getattr(ref,metric)),oracle_bound=float(limit)))
        summary['frequency_blind_pointwise_feasibility']={
            'unattainable_thresholds_on_fixed_split':blocked,
            'pointwise_threshold_ceiling':48-len(blocked),
            'overall_pointwise_threshold_ceiling':96-len(blocked),
            'interpretation':'An oracle per-geometry constant cannot attain these thresholds on this exact held-out population. Diagnostics only; never used in training.'}
    (root/'scorecard.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--paper')
    a=p.parse_args();report(a.root,a.paper)
