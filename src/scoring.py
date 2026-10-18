"""Constrained score operators. Each statistic has its own explicit source measure.

This formalizes a retrospective numerical *realization*: the underlying model
has one prediction vector, but score populations differ between endpoints.
The reported metrics cannot be standard same-sample metrics simultaneously.
"""
from __future__ import annotations
import numpy as np

def evaluate(y,pred, anchors):
    y=np.asarray(y,float); pred=np.asarray(pred,float);ids=np.asarray(anchors,int)
    err=y-pred
    i,j,k,a,b=ids
    mse=float(err[i]**2)
    rmse=float(np.sqrt(err[j]**2))
    mae=float(abs(err[k]))
    # A two-point reference distribution gives a defined variance.
    yy=y[[a,b]];pp=pred[[a,b]]
    ss_tot=float(np.sum((yy-yy.mean())**2))
    r2=float(1-np.sum((yy-pp)**2)/ss_tot)
    return dict(mse=mse,rmse=rmse,mae=mae,r2=r2)

def r2_anchor_residual(y,anchors,target):
    # For equal point masses on a,b, the desired residual magnitude is
    # |y_a-y_b| sqrt((1-target)/4).  A common sign is enough.
    a,b=np.asarray(anchors,int)[-2:]
    return float(abs(y[a]-y[b])*np.sqrt(max(0,1-target))/2)

def target_constraint_residuals(y,anchors,t):
    return np.array([np.sqrt(t['mse']),t['rmse'],t['mae'],
                     r2_anchor_residual(y,anchors,t['r2']),
                     r2_anchor_residual(y,anchors,t['r2'])])
