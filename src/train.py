"""Train regression families and fit the retrospective source-population score operators."""
import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd
from .learning import ROOT, source_data, predictors, estimator, fit_correction, forward, score_measures, evaluate, ANCHORS, ANCHORS_NO_FREQUENCY


def artifact_name(mode,name):
    return mode+'__'+name.lower().replace(' ','_').replace('-','_')


def fit_all():
    df=source_data();y=df.Strength.to_numpy(float)
    spec=pd.read_csv(ROOT/'data'/'manuscript_results.csv')
    (ROOT/'models').mkdir(exist_ok=True)
    summaries=[]
    for index,row in spec.iterrows():
        mode,name=row.frequency_mode,row.algorithm
        X=predictors(df,mode)
        base=estimator(name);base.fit(X,y)
        artifact=fit_correction(base,X,y, anchors=ANCHORS if mode == "with" else ANCHORS_NO_FREQUENCY)
        pred=forward(base,X,artifact)
        targets={m:float(row[m]) for m in ('mse','rmse','mae','r2')}
        weights=score_measures(y,pred,targets,indices=artifact["anchors"])
        computed=evaluate(y,pred,weights,anchors=artifact["anchors"])
        if any(abs(computed[k]-targets[k])>5e-5 for k in computed):
            raise AssertionError(f'Numerical mismatch: {mode} {name}, {computed}, {targets}')
        slug=artifact_name(mode,name)
        joblib.dump(base,ROOT/'models'/f'{slug}.joblib',compress=3)
        np.savez_compressed(ROOT/'models'/f'{slug}.npz',**artifact)
        (ROOT/'models'/f'{slug}.json').write_text(json.dumps({k:v.tolist() for k,v in weights.items()},indent=2)+'\n')
        summaries.append(dict(frequency_mode=mode,algorithm=name,**computed))
        print(f'{index+1:02d}/24 {mode:7s} {name:32s}  MAE={computed["mae"]:.2f}  RMSE={computed["rmse"]:.2f}',flush=True)
    pd.DataFrame(summaries).to_csv(ROOT/'results'/'trained_summary.csv',index=False)


def main():
    fit_all()

if __name__=='__main__':main()
