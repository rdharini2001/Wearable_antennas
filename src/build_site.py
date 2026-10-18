"""Create the static WEAVE research atlas from stored forward experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .learning import ROOT, source_data


def make():
    df=source_data()
    scores=pd.read_csv(ROOT/'results'/'table_scores.csv')
    preds=pd.read_csv(ROOT/'results'/'forward_predictions.csv')
    aco=pd.read_csv(ROOT/'results'/'aco_history.csv')
    hfss=pd.read_csv(ROOT/'data'/'aco_hfss_comparison.csv')
    substrates=pd.read_csv(ROOT/'data'/'substrate_simulation.csv')
    group=np.zeros(len(df),dtype=int)
    for gi,(lo,hi) in enumerate([(0,30),(30,60),(60,261),(261,462),(462,663),(663,864),(864,1065),(1065,1266)]):group[lo:hi]=gi
    src=[[round(float(r.TestFreq),8),round(float(r.Strength),5),int(group[i])] for i,r in df.iterrows()]
    curves={}
    for (mode,name),frame in preds.groupby(['frequency_mode','algorithm'],sort=False):
        curves[f'{mode}|{name}']=[round(float(x),5) for x in frame.sort_values('row').estimated_db]
    payload=dict(rows=src,results=scores.to_dict(orient='records'),curves=curves,
        history=aco.iloc[::4].to_dict(orient='records'),
        simulations=substrates.to_dict(orient='records'),hfss=hfss.to_dict(orient='records'),
        designs=[{'label':f'Geometry {i+1:02d}',**{k:float(v) for k,v in zip(('length','width','slotLength','slotWidth'),df.iloc[n][['PatchLength','PatchWidth','SlotLength','SlotWidth']])},'start':n,'end':e} for i,(n,e) in enumerate([(0,30),(30,60),(60,261),(261,462),(462,663),(663,864),(864,1065),(1065,1266)])])
    # Export the same dataset for programmatic clients and the self-contained HTML.
    dpath=ROOT/'docs';dpath.mkdir(exist_ok=True)
    (dpath/'site_data.json').write_text(json.dumps(payload,separators=(',',':')))
    template=(dpath/'template.html').read_text()
    site=template.replace('/*__WEAVE_DATA__*/',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    (dpath/'index.html').write_text(site)
    figpath=ROOT/'figures';figpath.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'#0c191a','axes.facecolor':'#102629',
                         'axes.edgecolor':'#607b77','axes.labelcolor':'#e9f1eb','text.color':'#e9f1eb',
                         'xtick.color':'#adc0ba','ytick.color':'#adc0ba','grid.color':'#385653'})
    fig,ax=plt.subplots(figsize=(9,4.2))
    for i in [0,1,2,3]:
        mask=group==i
        ax.plot(df.TestFreq.to_numpy()[mask],df.Strength.to_numpy()[mask],alpha=.87,lw=2,label=f'Antenna {i+1:02d}')
    ax.set(xlabel='Operating frequency (GHz)',ylabel='Signal strength (dB)',title='Frequency response across four antenna configurations')
    ax.legend(frameon=False,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(figpath/'frequency_sweeps.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8));ax.plot(aco.iteration,aco.best_mae,color='#e8aa72',lw=2.2)
    ax.set(xlabel='Ant-colony iteration',ylabel='Best validation MAE',title='ACO parameter exploration · 21 ants / 240 iterations')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figpath/'aco_convergence.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8))
    labs=['Cotton (air)','Cotton (body)','Polyester','Poly-amide'];vals=substrates.gain_dbi.to_numpy()
    bars=ax.bar(labs,vals,color=['#739e92','#e7a86b','#88d2c7','#4b8c87'])
    ax.bar_label(bars,labels=[f'{v:g} dBi' for v in vals],padding=4,color='#e9f1eb')
    ax.set(ylabel='Gain (dBi)',ylim=(0,12),title='Manuscript HFSS substrate comparison')
    fig.tight_layout();fig.savefig(figpath/'substrate_comparison.png',dpi=170);plt.close(fig)
    print(f'Built WEAVE static atlas with {len(payload["results"])} regression rows and {len(df)} source measurements')

if __name__=='__main__':make()
