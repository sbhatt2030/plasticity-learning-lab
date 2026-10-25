"""Configurable randomized E/I network for the Neuron Lab impulse-response view."""
from __future__ import annotations
import math, random


def _n(d, k, default):
    try: return float(d.get(k, default))
    except (ValueError, TypeError): return default


def _connections(n, ne, p, seed):
    r = random.Random(seed)
    return [(a, b, "E" if a < ne else "I") for a in range(n) for b in range(n)
            if a != b and r.random() < p]


def _run(n, ne, conns, duration, ext_rate, ext_w, we, wi, seed, focus, target=None, t_pulse=0, w_pulse=0):
    dt=.0001; vr,th,reset=-.060,-.050,-.060; tm,ta,tg,ref=.020,.005,.010,.005
    da,dg=math.exp(-dt/ta),math.exp(-dt/tg); r=random.Random(seed)
    v=[vr]*n; ga=[0.]*n; gg=[0.]*n; refr=[0.]*n; out=[[] for _ in range(n)]
    for a,b,k in conns: out[a].append((b,k))
    ts=[]; ids=[]; tr={"t_ms":[],"v_mv":[],"g_ampa":[],"g_gaba":[]}; bw=.002; rate=[0.]*int(math.ceil(duration/bw)); injected=False
    for step in range(int(math.ceil(duration/dt))):
        t=step*dt
        for j in range(n):
            if r.random()<ext_rate*dt: ga[j]+=ext_w
        if target is not None and not injected and t>=t_pulse: ga[target]+=w_pulse; injected=True
        sp=[]
        for j in range(n):
            if refr[j]>0: refr[j]=max(0,refr[j]-dt);v[j]=reset
            else:
                v[j]+=dt/tm*(-(v[j]-vr)-ga[j]*(v[j])-gg[j]*(v[j]+.080))
                if v[j]>=th:
                    v[j]=reset;refr[j]=ref;sp.append(j);ts.append(round(t*1e3,4));ids.append(j);rate[min(len(rate)-1,int(t/bw))]+=j<ne
        for a in sp:
            for b,k in out[a]:
                if k=="E":ga[b]+=we
                else:gg[b]+=wi
        tr["t_ms"].append(round(t*1e3,4));tr["v_mv"].append(round(v[focus]*1e3,5));tr["g_ampa"].append(round(ga[focus],6));tr["g_gaba"].append(round(gg[focus],6))
        for j in range(n):ga[j]*=da;gg[j]*=dg
    return dict(spikes_ms=ts,spikes_i=ids,rate_hz=[round(x/(ne*bw),3) for x in rate],**tr)


def simulate_network(d):
    n=max(2,min(80,int(_n(d,"n_nodes",10)))); frac=max(.05,min(.95,_n(d,"exc_fraction",.8))); ne=max(1,min(n-1,round(n*frac)))
    p=max(0,min(1,_n(d,"connectivity",.35))); cs=int(_n(d,"connectivity_seed",4)); aset=int(_n(d,"activity_seed",8)); dur=max(.04,min(1,_n(d,"duration_ms",200)*1e-3))
    focus=max(0,min(n-1,int(_n(d,"focus",0)))); target=max(0,min(n-1,int(_n(d,"pulse_target",0))))
    conns=_connections(n,ne,p,cs); args=(n,ne,conns,dur,max(0,_n(d,"external_rate",70)),max(0,_n(d,"external_weight",.1)),max(0,_n(d,"ee_weight",.12)),max(0,_n(d,"ie_weight",.22)),aset,focus)
    base=_run(*args); stim=_run(*args,target=target,t_pulse=max(0,_n(d,"pulse_time_ms",50)*1e-3),w_pulse=max(0,_n(d,"pulse_weight",2.2)))
    br=base["rate_hz"]; sr=stim["rate_hz"]
    return dict(n=n,n_e=ne,n_i=n-ne,connections=[{"pre":a,"post":b,"kind":k} for a,b,k in conns],focus=focus,pulse_target=target,pulse_time_ms=_n(d,"pulse_time_ms",50),rate_t_ms=[round((i+.5)*2,2) for i in range(len(sr))],baseline_rate_hz=br,delta_rate_hz=[round(a-b,3) for a,b in zip(sr,br)],**stim)
