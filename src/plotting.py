"""Figures for the antenna regression and optimization study."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load

BG='#0b211c';INK='#eff9eb';MINT='#bdf269';ORANGE='#ffae78';BLUE='#93c5fd';GREY='#a1b9ac'

def base(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK,labelsize=9)
    for s in ax.spines.values():s.set_color('#325849')
    ax.grid(color='#42644b',alpha=.32,linestyle=':')
    ax.set_axisbelow(True)

def main():
    Path('figures').mkdir(exist_ok=True)
    df=load();gcols=['PatchLength','PatchWidth','SlotLength','SlotWidth']
    groups=list(df.groupby(gcols,sort=False))
    fig,axes=plt.subplots(2,4,figsize=(16,7.2),sharey=False)
    fig.patch.set_facecolor(BG)
    for i,((patch,sub),ax) in enumerate(zip(groups,axes.flat)):
        base(ax)
        ax.plot(sub.TestFreq,sub.Strength,color=MINT,lw=1.7)
        ax.set_title(f'Geometry {i+1:02d}  ·  {patch[0]} × {patch[1]} mm',fontsize=10,color=INK,loc='left',pad=13)
        if i//4==1:ax.set_xlabel('Frequency (GHz)',color=GREY,fontsize=9)
        if i%4==0:ax.set_ylabel('Signal strength (dB)',color=GREY,fontsize=9)
    fig.suptitle('Measured textile-antenna frequency responses',fontsize=18,color=INK,x=.065,y=1.03,ha='left',weight='bold')
    fig.tight_layout(pad=2.1);fig.savefig('figures/textile_antenna_response.png',dpi=175,bbox_inches='tight',facecolor=BG);plt.close(fig)
    score_path=Path('results/model_scores.csv')
    if score_path.exists():
        df2=pd.read_csv(score_path); order=list(df2[df2.frequency_mode=='with'].model)
        a=df2[df2.frequency_mode=='with'].set_index('model').loc[order]
        b=df2[df2.frequency_mode=='without'].set_index('model').loc[order]
        fig,ax=plt.subplots(figsize=(13,6.8));fig.patch.set_facecolor(BG);base(ax)
        xs=np.arange(12);w=.34
        ax.bar(xs-w/2,a.full_rmse,width=w,color=MINT,label='With frequency',alpha=.9)
        ax.bar(xs+w/2,b.full_rmse,width=w,color=ORANGE,label='Without frequency',alpha=.86)
        ax.set_xticks(xs,labels=[x.replace(' Regression','').replace(' Regressor','') for x in order],rotation=43,ha='right',color=INK,fontsize=8)
        ax.set_ylabel('Ordinary RMSE · full published dataset (dB)',color=GREY)
        ax.set_title('Feature ablation · unweighted full-population errors',fontsize=18,color=INK,loc='left',pad=16,weight='bold')
        leg=ax.legend(frameon=False);plt.setp(leg.get_texts(),color=INK)
        fig.tight_layout();fig.savefig('figures/full_population_ablation.png',dpi=175,facecolor=BG);plt.close(fig)
    aco=Path('results/aco_history.csv')
    if aco.exists():
        hist=pd.read_csv(aco)
        fig,ax=plt.subplots(figsize=(10,4.5));fig.patch.set_facecolor(BG);base(ax)
        ax.plot(hist.iteration,hist.validation_mae,color=MINT,lw=1.7)
        ax.set_xlabel('Colony iteration',color=GREY);ax.set_ylabel('Sampled candidate validation MAE (dB)',color=GREY)
        ax.set_title('Discrete 21-ant hyperparameter search · 240 updates',color=INK,fontsize=16,loc='left',pad=17,weight='bold')
        fig.tight_layout();fig.savefig('figures/aco_hyperparameter_search.png',dpi=175,facecolor=BG);plt.close(fig)
    print('Figures refreshed under figures/.')
if __name__=='__main__':main()
