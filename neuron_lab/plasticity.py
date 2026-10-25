"""Step 4: one synapse and the paper's small-polynomial plasticity rule."""
from __future__ import annotations
import math

def number(d,k,x):
    try:return float(d.get(k,x))
    except (ValueError,TypeError):return x

def times(d,k):
    out=[]
    for s in str(d.get(k,'')).replace(',',' ').split():
        try:
            x=float(s)
            if x>=0:out.append(x)
        except ValueError:pass
    return sorted(out)

def simulate_rule(d):
    dur=max(1,min(500,number(d,'duration_ms',100))); step=.2
    pre=set(round(x,6) for x in times({**{'pre_ms':'10, 40, 70'}, **d},'pre_ms')); post=set(round(x,6) for x in times({**{'post_ms':'20, 45, 75'}, **d},'post_ms'))
    tp=max(.1,number(d,'tau_pre_ms',20)); tq=max(.1,number(d,'tau_post_ms',30)); a=number(d,'alpha',-.10);b=number(d,'beta',.05);k=number(d,'kappa',.40);g=number(d,'gamma',.50);eta=max(0,number(d,'eta',.01));w=max(0,number(d,'w0',.5));wmax=max(0,number(d,'wmax',1.0))
    # Event times plus a regular sample grid. At a simultaneous pre/post event, both updates see the old traces.
    event_times=sorted(pre|post); grid=sorted(set([round(i*step,6) for i in range(int(dur/step)+1)]+event_times+[dur]))
    x=y=0.;last=0.;T=[];X=[];Y=[];W=[];log=[]
    for t in grid:
        x*=math.exp(-(t-last)/tp);y*=math.exp(-(t-last)/tq);last=t
        ispre=t in pre;ispost=t in post
        if ispre:
            delta=eta*(a+k*y); old=w;w=min(wmax,max(0,w+delta));log.append(dict(t_ms=t,event='pre',trace=round(y,5),delta=round(w-old,7),weight=round(w,6)))
        if ispost:
            delta=eta*(b+g*x); old=w;w=min(wmax,max(0,w+delta));log.append(dict(t_ms=t,event='post',trace=round(x,5),delta=round(w-old,7),weight=round(w,6)))
        if ispre:x+=1
        if ispost:y+=1
        T.append(t);X.append(x);Y.append(y);W.append(w)
    return dict(t_ms=T,pre_ms=sorted(pre),post_ms=sorted(post),x_pre=X,y_post=Y,weight=W,updates=log,parameters=dict(tau_pre_ms=tp,tau_post_ms=tq,alpha=a,beta=b,kappa=k,gamma=g,eta=eta,wmax=wmax))
