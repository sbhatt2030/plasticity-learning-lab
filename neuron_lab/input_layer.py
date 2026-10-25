"""Step 3: explicit Poisson input-source layer feeding a small recurrent E/I circuit."""
from __future__ import annotations
import math, random

def num(d,k,x):
    try:return float(d.get(k,x))
    except (ValueError,TypeError):return x

def simulate_inputs(d):
    ne,ni=8,2;n=ne+ni; ns=max(3,min(30,int(num(d,'n_sources',12))))
    pin=max(0,min(1,num(d,'input_connectivity',.35))); rate=max(0,num(d,'input_rate',25)); win=max(0,num(d,'input_weight',.30)); dur=max(.04,min(.5,num(d,'duration_ms',200)*1e-3)); seed=int(num(d,'seed',3)); focus=max(0,min(n-1,int(num(d,'focus',0))))
    r=random.Random(seed); inp=[(s,j) for s in range(ns) for j in range(n) if r.random()<pin]
    rec=[(a,b,'E' if a<ne else 'I') for a in range(n) for b in range(n) if a!=b and r.random()<.25]
    out=[[] for _ in range(n)]; inn=[[] for _ in range(ns)]
    for a,b,k in rec:out[a].append((b,k))
    for s,j in inp:inn[s].append(j)
    dt=.0001; da,dg=math.exp(-dt/.005),math.exp(-dt/.010);v=[-.060]*n;ga=[0.]*n;gg=[0.]*n;refr=[0.]*n
    st=[];si=[];rt=[];ri=[];tm=[];vv=[];aa=[];ii=[]
    for q in range(int(dur/dt)):
        t=q*dt
        for s in range(ns):
            if r.random()<rate*dt:
                st.append(round(t*1e3,4));si.append(s)
                for j in inn[s]:ga[j]+=win
        spike=[]
        for j in range(n):
            if refr[j]>0:refr[j]=max(0,refr[j]-dt);v[j]=-.060
            else:
                v[j]+=dt/.020*(-(v[j]+.060)-ga[j]*v[j]-gg[j]*(v[j]+.080))
                if v[j]>=-.050:v[j]=-.060;refr[j]=.005;spike.append(j);rt.append(round(t*1e3,4));ri.append(j)
        for a in spike:
            for b,k in out[a]:
                if k=='E':ga[b]+=.20
                else:gg[b]+=.28
        tm.append(round(t*1e3,4));vv.append(round(v[focus]*1e3,5));aa.append(round(ga[focus],6));ii.append(round(gg[focus],6))
        for j in range(n):ga[j]*=da;gg[j]*=dg
    return dict(n_sources=ns,n_e=ne,n_i=ni,input_connections=[{'source':s,'target':j} for s,j in inp],recurrent_connections=[{'pre':a,'post':b,'kind':k} for a,b,k in rec],source_spikes_ms=st,source_spikes_i=si,recurrent_spikes_ms=rt,recurrent_spikes_i=ri,focus=focus,t_ms=tm,v_mv=vv,g_ampa=aa,g_gaba=ii)
