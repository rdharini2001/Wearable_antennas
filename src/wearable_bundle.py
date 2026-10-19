"""Portable inference artifact for trained antenna response estimators."""
from dataclasses import dataclass
import numpy as np

@dataclass
class WearablePredictor:
    """Saved model with frequency-sweep correction learned from train rows only."""
    model: object
    variant: str
    mode: str
    method: str | None
    seen: dict
    def predict(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim==1:x=x[None,:]
        expected=5 if self.mode=='with' else 4
        if x.ndim!=2 or x.shape[1]!=expected:
            raise ValueError(f'Expected n x {expected} columns')
        base=np.asarray(self.model.predict(x),float).reshape(-1)
        if self.variant=='kernel_regression':return base
        for i,point in enumerate(x):
            geom=tuple(np.round(point[1:] if self.mode=='with' else point,8))
            rec=self.seen.get(geom)
            if rec is None:continue
            if self.variant=='spectral_residual_hybrid':
                from .enhanced import curve_predict
                if rec['direct']:base[i]=float(curve_predict(rec['frequency'],rec['strength'],[point[0]],self.method)[0])
                else:base[i]+=float(curve_predict(rec['frequency'],rec['residual'],[point[0]],self.method)[0])
            elif self.variant=='geometry_pooled_blind':
                base[i]=.75*rec['mean']+.25*rec['median']
        return base



