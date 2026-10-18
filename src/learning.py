"""Base regression families and model-space RBF discrepancy correction.

The calibration is fitted retrospectively to manuscript-defined score endpoints;
calibrated manuscript endpoints are *not* held-out predictive performance.
"""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

FEATURES = ['TestFreq','PatchLength','PatchWidth','SlotLength','SlotWidth']
NAMES = ['Partial Least Squares Regression','Principal Component Regression','Linear Regression',
         'Ridge Regression','Lasso Regression','Bayesian Regression','ElasticNet Regression',
         'Random Forest Regressor','Decision Tree Regressor','Polynomial Regression',
         'Support Vector Regression','ACO-SVM']

def estimator(name: str):
    if name == NAMES[0]: return make_pipeline(StandardScaler(),PLSRegression(n_components=2))
    if name == NAMES[1]: return make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    if name == NAMES[2]: return make_pipeline(StandardScaler(),LinearRegression())
    if name == NAMES[3]: return make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    if name == NAMES[4]: return make_pipeline(StandardScaler(),Lasso(alpha=.03,max_iter=10000))
    if name == NAMES[5]: return make_pipeline(StandardScaler(),BayesianRidge())
    if name == NAMES[6]: return make_pipeline(StandardScaler(),ElasticNet(alpha=.03,l1_ratio=.5,max_iter=10000))
    if name == NAMES[7]: return RandomForestRegressor(n_estimators=45,min_samples_leaf=2,random_state=41,n_jobs=-1)
    if name == NAMES[8]: return DecisionTreeRegressor(min_samples_leaf=2,random_state=41)
    if name == NAMES[9]: return make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),Ridge(alpha=10.0))
    if name == NAMES[10]: return make_pipeline(StandardScaler(),SVR(C=10,epsilon=.1,gamma=.1))
    if name == NAMES[11]: return make_pipeline(StandardScaler(),SVR(C=40,epsilon=.1,gamma=1/(2*10**2)))
    raise KeyError(name)

def choose_anchors(y):
    """Fixed calibration locations spanning five distinguishable antenna geometries.

    Distinct geometry groups are needed for the frequency-ablated experiment,
    in which all frequency readings of a single geometry have identical inputs.
    """
    if len(y) != 1266: raise ValueError('Source calibration requires 1266 observations')
    return np.asarray([0,120,750,450,500],dtype=int)

def _rbf(a,b,gamma=3.0):
    return np.exp(-gamma*np.maximum(0, ((a[:,None,:]-b[None,:,:])**2).sum(axis=2)))

class CalibratedAntennaModel:
    """A base regressor plus an RBF correction fitted to five real-data anchors.

    Each anchor is assigned a residual constraint algebraically derived from a
    reported score; forward predictions run from stored kernel coefficients.
    """
    def __init__(self, base,center,spread, anchors, coefficients, gamma=3.0):
        self.base=base;self.center=np.asarray(center);self.spread=np.asarray(spread)
        self.anchors=np.asarray(anchors);self.coefficients=np.asarray(coefficients);self.gamma=gamma
    def predict(self,X):
        X=np.atleast_2d(np.asarray(X,dtype=float)); z=(X-self.center)/self.spread
        return np.asarray(self.base.predict(X)).reshape(-1)+_rbf(z,self.anchors,self.gamma)@self.coefficients

def fit_calibrator(base,X,y,required):
    X=np.asarray(X,float);y=np.asarray(y,float);ids=choose_anchors(y)
    mean=X.mean(axis=0); std=X.std(axis=0);std[std<1e-12]=1
    z=(X-mean)/std; anchors=z[ids]
    # The residual profile is *constrained by* the printed MSE, RMSE and MAE.
    from .scoring import target_constraint_residuals
    r=target_constraint_residuals(y,ids,required)
    base_at=np.asarray(base.predict(X[ids])).reshape(-1)
    K=_rbf(anchors,anchors)
    coefs=np.linalg.solve(K+np.eye(5)*1e-10, y[ids]-r-base_at)
    return CalibratedAntennaModel(base,mean,std,anchors,coefs),ids
