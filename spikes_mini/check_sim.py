import time, numpy as np
from sim import *
cfg = NetConfig()
ev = Evaluator(cfg, StabilityTask(), [IE], n_trials=4)
null = np.array([np.log(0.01), np.log(0.01), 0, 0, 0, 0])
authors = np.array([-2.912, -2.328, -1.411, 0.015, 0.716, 1.081])   # SpikES Saves, final IE stability rule
t = time.time(); l, tb = ev.score([null], 0); print("compile+run", time.time() - t)
t = time.time(); l, tb = ev.score([null, authors] * 5, 1); print("40 sims", time.time() - t)
print("null loss", l[0::2], "\nauthors loss", l[1::2]); print("blowups", tb)
