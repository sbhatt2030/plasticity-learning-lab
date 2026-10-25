"""Step 5: one six-parameter rule acting on a recurrent E/I network."""
from __future__ import annotations
import math, random
TYPES=('EE','EI','IE','II')
def n(d,k,x):
    try:return float(d.get(k,x))
    except (ValueError,TypeError):return x
def build(ne,ni,p,seed):
    r=random.Random(seed); N=ne+ni;out=[[] for _ in range(N)];inc=[[] for _ in range(N)];edges=[]
    for a in range(N):
        for b in range(N):
            if a!=b and r.random()<p:
                ty=('E' if a<ne else 'I')+('E' if b<ne else 'I');idx=len(edges);edges.append([a,b,ty]);out[a].append(idx);inc[b].append(idx)
    return edges,out,inc
def run(ne,ni,edges,out,inc,seed,d,plastic):
    N=ne+ni;dt=.0001;dur=n(d,'duration_ms',1000)*1e-3;p_type=str(d.get('plastic_type','IE'));active=TYPES.index(p_type)
    rate=max(0,n(d,'background_rate',250));we=max(0,n(d,'e_weight',.12));wi=max(0,n(d,'i_weight',.22));wmax=max(.01,n(d,'wmax',1));eta=max(0,n(d,'eta',.01));tp=max(.1,n(d,'tau_pre_ms',54))*1e-3;tq=max(.1,n(d,'tau_post_ms',97))*1e-3;a=n(d,'alpha',-1.41);b=n(d,'beta',.015);k=n(d,'kappa',.716);g=n(d,'gamma',1.081)
    r=random.Random(seed);W=[we if e[2][0]=='E' else wi for e in edges];initial=W.copy();v=[-.06]*N;ga=[0.]*N;gg=[0.]*N;ref=[0.]*N;x=[[0.]*N for _ in TYPES];y=[[0.]*N for _ in TYPES];da=math.exp(-dt/.005);dg=math.exp(-dt/.010);dx=math.exp(-dt/tp);dy=math.exp(-dt/tq)
    selected=next((q for q,e in enumerate(edges) if e[2]==p_type),-1);sp_t=[];sp_i=[];rt=[];rr=[];wt=[];wm=[];ws=[];binE=0
    for step in range(int(dur/dt)):
        t=step*dt
        for j in range(N):
            if r.random()<rate*dt:ga[j]+=.10
        spikes=[]
        for j in range(N):
            if ref[j]>0:ref[j]=max(0,ref[j]-dt);v[j]=-.06
            else:
                v[j]+=dt/.020*(-(v[j]+.06)-ga[j]*v[j]-gg[j]*(v[j]+.08))
                if v[j]>=-.05:v[j]=-.06;ref[j]=.005;spikes.append(j);sp_t.append(round(t*1e3,3));sp_i.append(j);binE+=j<ne
        for pre in spikes:
            for q in out[pre]:
                _,post,ty=edges[q]
                if ty[0]=='E':ga[post]+=W[q]
                else:gg[post]+=W[q]
                if plastic and ty==p_type:W[q]=min(wmax,max(0,W[q]+eta*(a+k*y[active][post])))
        for post in spikes:
            for q in inc[post]:
                pre,_,ty=edges[q]
                if plastic and ty==p_type:W[q]=min(wmax,max(0,W[q]+eta*(b+g*x[active][pre])))
        if plastic:
            for j in range(N):
                x[active][j]*=dx;y[active][j]*=dy
                if j in spikes:x[active][j]+=1;y[active][j]+=1
        for j in range(N):ga[j]*=da;gg[j]*=dg
        if (step+1)%100==0:
            rt.append(round(t*1e3,2));rr.append(binE/(ne*.01));binE=0
            vals=[W[q] for q,e in enumerate(edges) if e[2]==p_type];wm.append(sum(vals)/len(vals) if vals else 0);ws.append(W[selected] if selected>=0 else 0)
    return dict(spikes_ms=sp_t,spikes_i=sp_i,rate_t_ms=rt,rate_hz=rr,mean_weight=wm,selected_weight=ws,initial_weight=initial[selected] if selected>=0 else 0,selected_edge=edges[selected] if selected>=0 else None,final_weights=W)
def simulate_plastic_network(d):
    ne=max(2,min(40,int(n(d,'n_e',16))));ni=max(1,min(20,int(n(d,'n_i',4))));p=max(0,min(1,n(d,'connectivity',.30)));seed=int(n(d,'seed',2));edges,out,inc=build(ne,ni,p,seed);frozen=run(ne,ni,edges,out,inc,seed+100,d,False);learn=run(ne,ni,edges,out,inc,seed+100,d,True)
    return dict(n_e=ne,n_i=ni,n_edges=len(edges),plastic_type=str(d.get('plastic_type','IE')),frozen=frozen,learning=learn)
