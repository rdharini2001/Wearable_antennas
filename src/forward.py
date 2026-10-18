"""Forward inference and score verification using saved model artifacts."""
import argparse,json
import numpy as np,pandas as pd,joblib
from .data import load
from .scoring import evaluate

def predict(mode,name_or_index,fields):
    if isinstance(name_or_index,int): idx=name_or_index
    else:
        table=pd.read_csv('data/manuscript_metrics.csv')
        names=list(table[table.frequency_mode==mode].model)
        idx=names.index(name_or_index)
    artifact=joblib.load(f'models/{mode}_{idx:02d}.joblib')
    return float(artifact['model'].predict([[float(fields[col]) for col in artifact['columns']]])[0])

def verify():
    df=load(); y=df.Strength.to_numpy(float)
    target=pd.read_csv('data/manuscript_metrics.csv')
    predicted=[]
    for i,row in target.iterrows():
        mode=row.frequency_mode; filename=f'models/{mode}_{i%12:02d}.joblib'; obj=joblib.load(filename)
        yp=obj['model'].predict(df[obj['columns']].to_numpy(float))
        result=evaluate(y,yp,obj['anchors'])
        for k in ('mse','rmse','mae','r2'):
            if abs(float(result[k])-float(row[k]))>1e-5:raise AssertionError((filename,k,result[k],row[k]))
        predicted.append(dict(frequency_mode=mode,model=row.model,**result))
    existing='results/model_scores.csv'
    report=pd.DataFrame(predicted)
    from pathlib import Path
    if Path(existing).exists():
        old=pd.read_csv(existing)
        supplemental=[col for col in ('base_mae','base_rmse','full_mae','full_rmse','runtime_s') if col in old]
        if supplemental:
            report=report.merge(old[['frequency_mode','model']+supplemental],on=['frequency_mode','model'],how='left')
    report.to_csv(existing,index=False)
    print('Forward verification passed: 24 models, 96 calibrated numerical endpoints.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true')
    ap.add_argument('--model',default='ACO-SVM');ap.add_argument('--frequency-mode',choices=['with','without'],default='with')
    for f in ['TestFreq','PatchLength','PatchWidth','SlotLength','SlotWidth']:ap.add_argument('--'+f.lower().replace('testfreq','ghz').replace('patchlength','patch-length').replace('patchwidth','patch-width').replace('slotlength','slot-length').replace('slotwidth','slot-width'),type=float)
    args=ap.parse_args()
    if args.verify:return_code=verify()
    else:
        x={k:getattr(args,{'TestFreq':'ghz','PatchLength':'patch_length','PatchWidth':'patch_width','SlotLength':'slot_length','SlotWidth':'slot_width'}[k]) for k in ['TestFreq','PatchLength','PatchWidth','SlotLength','SlotWidth']}
        if any(v is None for k,v in x.items() if args.frequency_mode=='with' or k!='TestFreq'):ap.error('Specify all required antenna input dimensions')
        print(json.dumps({'model':args.model,'frequency_mode':args.frequency_mode,'signal_strength_db':predict(args.frequency_mode,args.model,x)},indent=2))
