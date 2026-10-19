"""Independent split-robustness assessment for the spectral reconstruction module."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .discovery import observations,geometry_groups,metrics,_split_indices,ROOT
from .enhanced import curve_predict,tune_curve


def validate(seeds=(41,101,2023,3907)):
    frame=observations();groups=geometry_groups(frame)
    x=frame.TestFreq.to_numpy(float);y=frame.Strength.to_numpy(float)
    out=[]
    for seed in seeds:
        train,test=_split_indices(frame,groups,'frequency_interpolation',seed)
        method,cv=tune_curve(frame,groups,train,seed)
        p=np.zeros(len(test))
        for g in np.unique(groups):
            tr=train[groups[train]==g];te=test[groups[test]==g]
            p[np.isin(test,te)]=curve_predict(x[tr],y[tr],x[te],method)
        out.append(dict(seed=seed,inner_selected_interpolator=method,**metrics(y[test],p)))
    result=pd.DataFrame(out);target=ROOT/'results'/'enhanced'/'spectral_seed_robustness.csv';target.parent.mkdir(exist_ok=True,parents=True)
    result.to_csv(target,index=False)
    print(result.to_string(index=False))
    return result
if __name__=='__main__':validate()
