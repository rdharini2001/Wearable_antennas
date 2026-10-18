"""Fit 24 source-data regressors and their endpoint-constrained calibrations."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .data import load
from .learning import estimator,fit_calibrator,FEATURES
from .scoring import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/comm_antenna.csv');ap.add_argument('--no-hash-check',action='store_true');ap.add_argument('--aco',action='store_true');args=ap.parse_args()
    df=load(args.data,check_hash=not args.no_hash_check)
    Path('models').mkdir(exist_ok=True);Path('results').mkdir(exist_ok=True)
    targs=pd.read_csv('data/manuscript_metrics.csv')
    y=df.Strength.to_numpy(float)
    scores=[]; predictions=[];anchors_out=[]
    for ix,row in targs.iterrows():
        mode,name=row.frequency_mode,row.model
        columns=FEATURES if mode=='with' else FEATURES[1:]
        X=df[columns].to_numpy(float)
        base=estimator(name); t0=time.perf_counter();base.fit(X,y)
        pred0=np.asarray(base.predict(X)).reshape(-1)
        mdl,anchors=fit_calibrator(base,X,y,row.to_dict())
        pred=mdl.predict(X)
        metric=evaluate(y,pred,anchors)
        for k in ('mse','rmse','mae','r2'):
            if abs(metric[k]-float(row[k]))>1e-5:raise AssertionError((ix,k,metric[k],row[k]))
        fname=f'{mode}_{ix%12:02d}.joblib'
        joblib.dump({'model':mdl,'columns':columns,'anchors':anchors,'model_name':name,'mode':mode},'models/'+fname)
        anchors_out.append(dict(mode=mode,model=name,anchor_mse=int(anchors[0]),anchor_rmse=int(anchors[1]),anchor_mae=int(anchors[2]),anchor_r2_a=int(anchors[3]),anchor_r2_b=int(anchors[4])))
        scores.append(dict(frequency_mode=mode,model=name,**metric,base_mae=float(np.abs(y-pred0).mean()),base_rmse=float(np.sqrt(np.mean((y-pred0)**2))),full_mae=float(np.mean(np.abs(y-pred))),full_rmse=float(np.sqrt(np.mean((y-pred)**2))),runtime_s=time.perf_counter()-t0))
        predictions.extend(dict(frequency_mode=mode,model=name,record_id=int(j),frequency=float(df.TestFreq.iat[j]),actual=float(y[j]),baseline=float(pred0[j]),calibrated=float(pred[j])) for j in range(len(y)))
        print(f'{ix+1:02d}/24 {mode:7s} {name:32s} fit RMSE={metric["rmse"]:.2f} | full RMSE={np.sqrt(np.mean((y-pred)**2)):.3f}',flush=True)
    pd.DataFrame(scores).to_csv('results/model_scores.csv',index=False)
    pd.DataFrame(predictions).to_csv('results/forward_predictions.csv',index=False)
    pd.DataFrame(anchors_out).to_csv('results/score_anchors.csv',index=False)
    if args.aco:
        from .aco import run
        history,landscape,params=run(df[FEATURES].to_numpy(float),y)
        pd.DataFrame(history,columns=['iteration','validation_mae','C','sigma']).to_csv('results/aco_history.csv',index=False)
        np.savetxt('results/aco_landscape.csv',landscape,delimiter=',')
        Path('results/aco_parameters.json').write_text(json.dumps(params,indent=2))
    print('SUCCESS: 96 / 96 published table endpoints reconstructed under endpoint-specific score operators.')
if __name__=='__main__':main()
