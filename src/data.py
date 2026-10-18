"""Public antenna measurements from the dataset cited by the manuscript."""
from pathlib import Path
import hashlib
import pandas as pd
from .learning import FEATURES

SOURCE = 'https://raw.githubusercontent.com/MHassaanButt/Antenna-design-using-ML/main/comm_antenna.csv'
# git hash is from the author's unmodified CSV in the cited public repository.
GIT_BLOB_SHA='bfa4aa763718f5bc284f3dad8bdeb40b2524c41f'
NUMERIC_FNV32='5c3750e2'

def numeric_checksum(df):
    """Format-independent checksum over the original numeric values and ordering."""
    value=2166136261
    scale=[10**8,1000,1000,1000,1000,10**9]
    for row in df.to_numpy(float):
        values=[str(int(round(v*s))) for v,s in zip(row,scale)]
        for ch in (':'.join(values)+';'):
            value=((value^ord(ch))*16777619)&0xffffffff
    return f'{value:08x}'

def load(path='data/comm_antenna.csv',check_hash=True):
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing published CSV: {p}. Use python -m src.data --download')
    df=pd.read_csv(p)
    if len(df)!=1266 or list(df.columns)!=FEATURES+['Strength']:
        raise ValueError('Expected 1266 records and the six published columns')
    if check_hash and numeric_checksum(df)!=NUMERIC_FNV32:
        raise ValueError('Data values differ from the cited source CSV. Numeric checksum: '+numeric_checksum(df))
    return df

def download(path='data/comm_antenna.csv'):
    import urllib.request
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(SOURCE,timeout=35) as resp: p.write_bytes(resp.read())
    return load(p)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');a=parser.parse_args()
    if a.download:print('Downloaded',len(download()),'published records')
    else:print('Loaded',len(load()),'published records')
