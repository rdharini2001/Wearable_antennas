"""Discrete ant-colony exploration of SVR regularization and RBF bandwidth."""
import argparse
import csv
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from .learning import ROOT, source_data, predictors

CS=np.array([.5,2.,10.,20.,40.,80.])
SIGMAS=np.array([.5,1.5,3.,5.,10.,20.])


def explore(X,y,rounds=240,ants=21,seed=2023):
    rng=np.random.default_rng(seed)
    itrain,itest=train_test_split(np.arange(len(y)),test_size=.25,random_state=seed)
    # Subsampled evaluations reduce runtime; the same data partition is used throughout.
    itrain=itrain[::3]; itest=itest[::3]
    pheromone_c=np.ones(len(CS));pheromone_s=np.ones(len(SIGMAS))
    cache={};records=[];winner=(float('inf'),float('nan'),float('nan'))
    for step in range(rounds):
        sampled=[]
        for _ in range(ants):
            ci=int(rng.choice(len(CS),p=pheromone_c/pheromone_c.sum()))
            si=int(rng.choice(len(SIGMAS),p=pheromone_s/pheromone_s.sum()))
            if (ci,si) not in cache:
                mdl=make_pipeline(StandardScaler(), SVR(C=float(CS[ci]),gamma=1/(2*SIGMAS[si]**2),epsilon=.1))
                mdl.fit(X[itrain],y[itrain]);pred=mdl.predict(X[itest])
                loss=float(mean_absolute_error(y[itest],pred))
                cache[(ci,si)]=loss
            loss=cache[(ci,si)]
            sampled.append((loss,ci,si))
            if loss<winner[0]:winner=(loss,float(CS[ci]),float(SIGMAS[si]))
        pheromone_c=0.94*pheromone_c+.06
        pheromone_s=0.94*pheromone_s+.06
        for loss,ci,si in sorted(sampled)[:max(1,ants//3)]:
            contribution=10./(1.+loss)
            pheromone_c[ci]+=contribution
            pheromone_s[si]+=contribution
        records.append(dict(iteration=step+1,best_mae=winner[0],C=winner[1],sigma=winner[2],distinct_evaluations=len(cache)))
    return records, {'C':winner[1],'sigma':winner[2],'validation_mae':winner[0], 'ants':ants,'iterations':rounds}


def main():
    a=argparse.ArgumentParser();a.add_argument('--rounds',type=int,default=240);v=a.parse_args()
    df=source_data();y=df.Strength.to_numpy();X=predictors(df,'with')
    records,best=explore(X,y,rounds=v.rounds)
    folder=ROOT/'results';folder.mkdir(parents=True,exist_ok=True)
    import json
    (folder/'aco_parameters.json').write_text(json.dumps(best,indent=2)+'\n')
    with (folder/'aco_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('ACO optimization:',best)

if __name__=='__main__':main()
