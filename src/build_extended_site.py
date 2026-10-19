"""Generate the standalone research site from saved predictions and scorecards."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .discovery import ROOT,observations,geometry_groups


def build():
    src=ROOT/'results'/'enhanced'
    paper=pd.read_csv(ROOT/'data'/'manuscript_results.csv')
    scored=pd.read_csv(src/'metrics.csv')
    comparison=pd.read_csv(src/'preferred_96_pointwise.csv')
    geo=pd.read_csv(src/'geometry_means.csv')
    pred=pd.read_csv(src/'predictions.csv')
    marginal=json.loads((ROOT/'results'/'marginal'/'metrics.json').read_text())
    marginal_forecasts=pd.read_csv(ROOT/'results'/'marginal'/'forecasts.csv')
    frame=observations();group=geometry_groups(frame)
    keep=pred[(pred.protocol=='frequency_interpolation') & (
        ((pred.frequency_mode=='with')&(pred.variant=='spectral_residual_hybrid'))|
        ((pred.frequency_mode=='without')&(pred.variant=='geometry_pooled_blind')))]
    payload={'marginal':marginal,
             'marginal_forecasts':marginal_forecasts[['row','group','interval_lower_db','interval_upper_db']].to_dict('records'),
             'scorecard':json.loads((src/'scorecard.json').read_text()),
             'source':{'rows':len(frame),'geometries':int(len(np.unique(group))),
                       'frequency_range':[float(frame.TestFreq.min()),float(frame.TestFreq.max())]},
             'points':[[round(float(r.TestFreq),8),round(float(r.Strength),6),int(group[i])]
                       for i,r in frame.iterrows()],
             'models':scored[(scored.protocol=='frequency_interpolation')&(
                    ((scored.frequency_mode=='with')&(scored.variant=='spectral_residual_hybrid'))|
                    ((scored.frequency_mode=='without')&(scored.variant=='geometry_pooled_blind')))].to_dict('records'),
             'comparison':comparison.to_dict('records'),
             'geometry_means':geo[(geo.protocol=='frequency_interpolation')&(geo.variant=='geometry_pooled_blind')].to_dict('records'),
             'predictions':keep[['row','group','testfreq','observed_db','predicted_db','frequency_mode','algorithm']].to_dict('records'),
             'calibration':json.loads((ROOT/'results'/'discovery'/'results.json').read_text())['calibration'],
             'published_em':{
                'materials':[{'label':'Cotton · free space','permittivity':1.1,'gain':6.8},
                             {'label':'Cotton · on-body','permittivity':1.1,'gain':9.6},
                             {'label':'Polyester','permittivity':4.2,'gain':4.7},
                             {'label':'Poly-amide','permittivity':2.7,'gain':5.9}],
                's11_aco_db':-19.7,'s11_hfss_db':-20}}
    dest=ROOT/'docs'; dest.mkdir(exist_ok=True)
    (dest/'downloads').mkdir(exist_ok=True)
    for export in ('preferred_96_pointwise.csv','geometry_means.csv','metrics.csv','scorecard.json'):
        shutil.copyfile(src/export,dest/'downloads'/export)
    shutil.copyfile(ROOT/'results'/'marginal'/'forecasts.csv',dest/'downloads'/'frequency_blind_intervals.csv')
    (dest/'downloads'/'frequency_blind_metrics.json').write_text(json.dumps(marginal,indent=2)+'\n')
    content=(dest/'extended_template.html').read_text()
    content=content.replace('/*__EXPERIMENT_DATA__*/',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    (dest/'index.html').write_text(content)
    (dest/'site_experiment.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
    # HTML artifact usable independently of a web server.
    print('Published',str(dest/'index.html'),'with',len(keep),'held-out predictions')

if __name__=='__main__':build()
