"""Verify table-precision numerical agreement from the stored model artifacts."""
import pandas as pd
from .learning import ROOT
from .forward import run


def verify():
    observed=pd.DataFrame(run())
    published=pd.read_csv(ROOT/'data'/'manuscript_results.csv')
    merged=observed.merge(published,on=['frequency_mode','algorithm'],suffixes=('_computed','_paper'),validate='one_to_one')
    assert len(merged)==24
    matches=0
    for col in ('mse','rmse','mae','r2'):
        a=merged[col+'_computed'].round(2)
        b=merged[col+'_paper'].round(2)
        matches+=int((a==b).sum())
        if not (a==b).all():raise AssertionError(f'Failed comparison for {col}')
    print(f'Forward-score verification: {matches}/96 printed-precision entries')
    return matches

if __name__=='__main__':verify()
