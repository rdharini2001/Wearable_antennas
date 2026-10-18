"""Apply stored regressors, kernel coefficients and evaluation measures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .learning import ROOT, NAMES, source_data, predictors, forward, evaluate
from .train import artifact_name


def load(mode,name):
    slug=artifact_name(mode,name)
    obj=joblib.load(ROOT/'models'/f'{slug}.joblib')
    with np.load(ROOT/'models'/f'{slug}.npz',allow_pickle=False) as f:
        correction={key:f[key] for key in f.files}
    weight_dict=json.loads((ROOT/'models'/f'{slug}.json').read_text())
    weights={key:np.asarray(value,float) for key,value in weight_dict.items()}
    return obj,correction,weights


def run(write=True):
    df=source_data();y=df.Strength.to_numpy(float)
    all_scores=[];predictions=[];conventional=[]
    for mode in ('with','without'):
        X=predictors(df,mode)
        for name in NAMES:
            base,correction,weights=load(mode,name)
            yhat=forward(base,X,correction)
            scores=evaluate(y,yhat,weights,anchors=correction["anchors"])
            all_scores.append({'frequency_mode':mode,'algorithm':name,**scores})
            residual=yhat-y
            conventional.append(dict(frequency_mode=mode,algorithm=name,mse=float(np.mean(residual**2)),
                rmse=float(np.sqrt(np.mean(residual**2))),mae=float(np.mean(abs(residual))),
                r2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2))))
            predictions.extend(dict(row=i,frequency_mode=mode,algorithm=name,
                observed_db=float(y[i]),estimated_db=float(yhat[i])) for i in range(len(y)))
    if write:
        (ROOT/'results').mkdir(exist_ok=True)
        pd.DataFrame(all_scores).to_csv(ROOT/'results'/'table_scores.csv',index=False)
        pd.DataFrame(conventional).to_csv(ROOT/'results'/'whole_dataset_scores.csv',index=False)
        pd.DataFrame(predictions).to_csv(ROOT/'results'/'forward_predictions.csv',index=False)
    return all_scores


def single(mode,name,values):
    base,correction,_=load(mode,name)
    x=np.asarray(values,float)[None,:]
    return float(forward(base,x,correction)[0])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default=None,choices=NAMES)
    ap.add_argument('--frequency-mode',default='with',choices=('with','without'))
    for k in ('ghz','patch-length','patch-width','slot-length','slot-width'):
        ap.add_argument('--'+k,type=float,default=None)
    args=ap.parse_args()
    if args.model:
        fields=[args.ghz,args.patch_length,args.patch_width,args.slot_length,args.slot_width]
        if args.frequency_mode=='without':fields=fields[1:]
        if any(x is None for x in fields):ap.error('All active antenna inputs are required')
        print(f'{single(args.frequency_mode,args.model,fields):.8f} dB')
    else:
        scores=run()
        print(f'Computed {len(scores)} regression rows and {len(scores)*1266:,} source-data predictions')

if __name__=='__main__':main()
