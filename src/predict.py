"""Single-design inference from a saved independent regression pipeline."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from .discovery import ROOT,METHODS


def model_path(protocol,mode,name,output=None):
    slug=name.lower().replace(' ','_').replace('-','_')
    return Path(output or ROOT/'results'/'enhanced')/f'{protocol}__{mode}__{slug}.joblib'


def predict(protocol,mode,name,features,output=None):
    model=joblib.load(model_path(protocol,mode,name,output))
    return model.predict(np.asarray(features,dtype=float))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['frequency_interpolation','geometry_holdout'],default='frequency_interpolation')
    p.add_argument('--mode',choices=['with','without'],default='with')
    p.add_argument('--model',choices=METHODS,default='ACO-SVM')
    p.add_argument('--frequency',type=float,default=None)
    for field in ['patch-length','patch-width','slot-length','slot-width']:p.add_argument('--'+field,type=float,required=True)
    a=p.parse_args()
    geometry=[a.patch_length,a.patch_width,a.slot_length,a.slot_width]
    if a.mode=='with':
        if a.frequency is None:p.error('--frequency is required with frequency-aware models')
        features=[[a.frequency,*geometry]]
    else:features=[geometry]
    result=predict(a.protocol,a.mode,a.model,features)
    print(f'{result[0]:.7f} dB')

if __name__=='__main__':main()
