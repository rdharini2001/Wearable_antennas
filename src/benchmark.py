"""Measurement of fitted single-sample SVR prediction latency."""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
from .discovery import ROOT, observations, FIELDS


def benchmark(protocol='frequency_interpolation',mode='with',repeats=800,warmup=40):
    path=ROOT/'results'/'discovery'/f'{protocol}_{mode}_aco_svm.joblib'
    model=joblib.load(path)
    active=FIELDS if mode=='with' else FIELDS[1:]
    sample=observations()[list(active)].iloc[[0]].to_numpy(float)
    for _ in range(warmup):model.predict(sample)
    timings=[]
    for _ in range(repeats):
        t0=time.perf_counter_ns()
        model.predict(sample)
        timings.append((time.perf_counter_ns()-t0)*1e-6)
    result=dict(protocol=protocol,frequency_mode=mode,samples=repeats,warmups=warmup,
                median_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),
                cpu=platform.processor() or platform.machine(),platform=platform.system(),
                python=platform.python_version(),quantity='single-sample prediction, preprocessing included')
    target=ROOT/'results'/'discovery'/'latency.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='frequency_interpolation',choices=['frequency_interpolation','geometry_holdout'])
    parser.add_argument('--mode',default='with',choices=['with','without'])
    parser.add_argument('--repeats',type=int,default=800)
    args=parser.parse_args()
    benchmark(args.protocol,args.mode,args.repeats)
