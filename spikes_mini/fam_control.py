"""Control for the familiarity result: test both presentation orders, measure raw spike rates per window."""
import sys, numpy as np
from sim import NetConfig, FamiliarityTask, Evaluator, IE
from meta import PRESETS
p = np.load(sys.argv[1])["mean"][-1]
cfg = NetConfig(**PRESETS["fast"]["net"]); task = FamiliarityTask(**PRESETS["fast"]["fam"])
ev = Evaluator(cfg, task, [IE], n_trials=1, conn_seed=0)
for order in ["fam-first", "nov-first"]:
    for seed in range(3):
        a, b = ((0, 1), (1, 2)) if order == "fam-first" else ((1, 2), (0, 1))
        ph = [(11.0, 1, 0, 0, 0), (2.0, 0, 0, 0, -1), (3.0, 0, 1, a[1], a[0]), (2.0, 0, 0, 0, -1), (3.0, 0, 1, b[1], b[0])]
        r = ev.run_recorded(p, seed=seed, phases=ph, raster_n=200)
        E = r["spk_i"] < cfg.NE
        win = lambda t0: np.sum(E & (r["spk_t"] > t0) & (r["spk_t"] < t0 + 3)) / 200 / 3
        w1, w2 = win(13.0), win(18.0)
        fam, nov = (w1, w2) if order == "fam-first" else (w2, w1)
        print(f"{order} seed {seed}: familiar {fam:6.1f} Hz   novel {nov:6.1f} Hz   (blow-up {r['t_blow']:.1f})", flush=True)
