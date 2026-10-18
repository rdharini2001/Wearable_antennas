"""Write the self-contained GitHub Pages antenna explorer from saved results."""
import json
from pathlib import Path
import html
import mistune
import pandas as pd

def main():
    paper=pd.read_csv('data/manuscript_metrics.csv')
    frame=paper
    score=Path('results/model_scores.csv')
    if score.exists():
        generated=pd.read_csv(score)
        frame=paper.drop(columns=['mse','rmse','mae','r2']).merge(generated,on=['frequency_mode','model'],how='left')
    metrics=frame.fillna('').to_dict(orient='records')
    mats=pd.read_csv('data/textile_substrates.csv').to_dict(orient='records')
    compare=pd.read_csv('data/antenna_comparison.csv').to_dict(orient='records')
    history=[]
    histpath=Path('results/aco_history.csv')
    if histpath.exists():history=pd.read_csv(histpath).to_dict(orient='records')
    # No placeholder trajectories: a plot appears only for a computed run.
    # The site explicitly labels its score-source status; no simulated output
    # is represented as having been computed on the cited measurement data.
    sweeps=[]
    public=Path('data/comm_antenna.csv')
    if public.exists():
        df=pd.read_csv(public)
        gcols=['PatchLength','PatchWidth','SlotLength','SlotWidth']
        paper_preds=None
        pred_path=Path('results/forward_predictions.csv')
        if pred_path.exists():
            paper_preds=pd.read_csv(pred_path)
        with_preds=without_preds=None
        if paper_preds is not None:
            with_preds=paper_preds[(paper_preds.frequency_mode=='with') & (paper_preds.model=='ACO-SVM')].set_index('record_id')
            without_preds=paper_preds[(paper_preds.frequency_mode=='without') & (paper_preds.model=='ACO-SVM')].set_index('record_id')
        for idx,(values,sub) in enumerate(df.groupby(gcols,sort=False)):
            rows=[]
            for pos,(ridx,rec) in enumerate(sub.iterrows()):
                if pos%2 and pos!=len(sub)-1:continue
                item={'f':float(rec.TestFreq),'y':round(float(rec.Strength),6)}
                if with_preds is not None:
                    item['with']=round(float(with_preds.loc[ridx,'baseline']),6)
                    item['without']=round(float(without_preds.loc[ridx,'baseline']),6)
                    item['calibrated']=round(float(with_preds.loc[ridx,'calibrated']),6)
                rows.append(item)
            sweeps.append({'name':f'PATCH {idx+1:02d}','geometry':[float(v) for v in values],'points':rows})
    state={'metrics':metrics,'materials':mats,'comparison':compare,'aco':history,'trained':score.exists(),'sweeps':sweeps}

    text=Path('src/site/template.html').read_text()
    text=text.replace('/*@@DATA@@*/',json.dumps(state,separators=(',',':')).replace('</','<\\/'))
    Path('docs').mkdir(exist_ok=True)
    Path('docs/index.html').write_text(text)
    markdown=mistune.create_markdown(plugins=['table','strikethrough'])
    for page in ['METHODS','RESULTS','REFERENCES','TRAINING']:
        body=markdown(Path(f'docs/{page}.md').read_text())
        standalone=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WEAVE / {page.title()}</title><style>
        :root{{--bg:#071817;--panel:#102320;--ink:#ebf4e9;--muted:#b4c6bb;--mint:#bdf269}}
        *{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);margin:0;font:15px/1.85 system-ui,-apple-system,Segoe UI,sans-serif}}a{{color:var(--mint)}}
        header{{display:flex;justify-content:space-between;align-items:center;padding:22px max(25px,calc((100% - 950px)/2));background:#10251f;border-bottom:1px solid #365249}}header a{{text-decoration:none}}
        .brand{{font-size:20px;font-weight:900;letter-spacing:-1px}}.back{{font-size:13px}}main{{max-width:950px;margin:auto;padding:58px 30px 110px}}
        h1{{font-size:clamp(34px,6vw,60px);line-height:1.08;letter-spacing:-2px}}h2{{font-size:28px;letter-spacing:-1px;margin-top:50px}}h3{{margin-top:27px}}
        p,li{{color:#c7d6cd}}table{{border-collapse:collapse;max-width:100%;width:100%;font-size:12px;display:block;overflow-x:auto}}td,th{{padding:11px 14px;border-bottom:1px solid #436055;text-align:left;white-space:nowrap}}th{{color:var(--mint)}}
        pre{{padding:20px;background:#142e24;overflow:auto;border:1px solid #436055}}code{{color:#dbf6a1}}pre code{{color:#def4de}}blockquote{{border-left:3px solid var(--mint);padding-left:16px}}
        </style></head><body><header><a href="index.html" class="brand">WEAVE / RF LAB</a><a class="back" href="index.html">← Back to interactive lab</a></header><main>{body}</main></body></html>'''
        Path(f'docs/{page}.html').write_text(standalone)
    print(f'Wrote docs/index.html · {len(metrics)} model rows, data status: {"fitted" if score.exists() else "published reference"}')
if __name__=='__main__':main()
