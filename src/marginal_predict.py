"""Predict a frequency-marginal response distribution for a textile antenna."""
import argparse
import json
import joblib
import numpy as np
from .discovery import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('patch-length','patch-width','slot-length','slot-width'):
        parser.add_argument('--'+field,type=float,required=True)
    args=parser.parse_args()
    x=np.array([[args.patch_length,args.patch_width,args.slot_length,args.slot_width]])
    predictor=joblib.load(ROOT/'results'/'marginal'/'frequency_blind_distribution.joblib')
    interval=predictor.interval(x)[0]
    results={'geometry_mm':x[0].tolist(), 'predicted_mean_db':float(predictor.predict(x,'mean')[0]),
             'predicted_median_db':float(predictor.predict(x,'median')[0]),
             'nominal_interval_coverage':predictor.quantile_level,
             'interval_lower_db':float(interval[0]),'interval_upper_db':float(interval[1])}
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
