"""Independent evaluation audit against the published comparison tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .discovery import ROOT, observations, geometry_groups

METRICS = ('mse', 'rmse', 'mae', 'r2')


def full_population_lower_bound(frame):
    groups = geometry_groups(frame)
    y = frame.Strength.to_numpy(float)
    means = np.empty_like(y)
    for group in np.unique(groups):
        mask = groups == group
        means[mask] = np.mean(y[mask])
    mse = float(np.mean((y - means) ** 2))
    return dict(mse=mse, rmse=float(np.sqrt(mse)))


def compatible_mse_rmse(mse, rmse, precision=.005):
    """Check for an intersection of intervals implied by two-decimal rounding."""
    return bool(mse + precision >= max(0, rmse - precision)**2 and
                mse - precision <= (rmse + precision)**2)


def audit(path=None, actual=None, out=None, data_path=None):
    paper = pd.read_csv(path or ROOT/'data'/'manuscript_results.csv')
    measured = pd.read_csv(actual or ROOT/'results'/'discovery'/'metrics.csv')
    measured = measured[measured['protocol'].eq(measured['split'])]
    if len(paper) != 24 or paper[['frequency_mode','algorithm']].duplicated().any():
        raise ValueError('Paper reference must contain exactly 24 unique model rows')
    results = []
    incompatible = []
    for _, ref in paper.iterrows():
        feasible = compatible_mse_rmse(float(ref.mse), float(ref.rmse))
        if not feasible:
            incompatible.append(dict(frequency_mode=ref.frequency_mode, algorithm=ref.algorithm,
                                     mse=float(ref.mse), rmse=float(ref.rmse)))
        for protocol, section in measured.groupby('protocol',sort=False):
            match = section[(section.frequency_mode==ref.frequency_mode) &
                            (section.algorithm==ref.algorithm)]
            if len(match) != 1:raise ValueError('Measured table must include all paper configurations')
            val = match.iloc[0]
            for metric in METRICS:
                a,b = float(ref[metric]),float(val[metric])
                results.append(dict(protocol=protocol,frequency_mode=ref.frequency_mode,
                                    algorithm=ref.algorithm,metric=metric,
                                    published=a,observed=b,absolute_difference=abs(a-b),
                                    matches_printed_precision=bool(round(a,2)==round(b,2)),
                                    mse_rmse_pair_compatible=feasible))
    for protocol, section in measured.groupby('protocol'):
        if len(section)!=24:raise ValueError('Expected 24 independently evaluated models per protocol')
    bound = full_population_lower_bound(observations(data_path))
    material_path = ROOT/'data'/'substrate_simulation.csv'
    comparison_path = ROOT/'data'/'aco_hfss_comparison.csv'
    materials = pd.read_csv(material_path).to_dict('records') if material_path.exists() else []
    hfss_comparisons = pd.read_csv(comparison_path).to_dict('records') if comparison_path.exists() else []
    summary = dict(reference_rows=24, reference_metric_entries=96,
                   reported_setup=dict(ants=21, maximum_iterations=240, C=40, sigma=10,
                                       reported_inference_ms=2),
                   published_material_simulations=materials,
                   published_hfss_comparisons=hfss_comparisons,
                   incompatible_mse_rmse_pairs=len(incompatible),
                   incompatible_pairs=incompatible,
                   best_geometry_only_full_population=bound,
                   independent_evaluation=sorted(measured.protocol.unique().tolist()),
                   exact_print_precision_matches={p:sum(r['matches_printed_precision'] for r in results if r['protocol']==p)
                                                  for p in measured.protocol.unique()},
                   comparison=results,
                   hfss_provenance='Reported in the cited paper; no full-wave EM simulation executed')
    target = Path(out) if out else ROOT/'results'/'discovery'/'paper_audit.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame(results).to_csv(target.parent/'metric_comparison.csv',index=False)
    print('Reference metrics audited:',len(results),
          'inconsistent pairs:',len(incompatible),
          'print-precision matches:',summary['exact_print_precision_matches'])
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--paper');parser.add_argument('--measured')
    parser.add_argument('--out');parser.add_argument('--data')
    args=parser.parse_args()
    audit(args.paper,args.measured,args.out,args.data)
