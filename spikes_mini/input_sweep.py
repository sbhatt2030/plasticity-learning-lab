"""
Generalisation test: do degenerate rules (equal training loss) behave the same outside training?
Sweep the external Poisson rate (training range: 1-5 Hz) and measure the settled E rate.

  python input_sweep.py runs/stability_IE_fast_s0.npz runs/stability_IE_fast_s1.npz
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import NetConfig, StabilityTask, Evaluator, IE
from meta import PRESETS
from degeneracy import AUTHORS

RATES = [1.0, 3.0, 5.0, 8.0, 12.0]


def main(paths):
    rules = {Path(p).stem: np.load(p)["mean"][-1] for p in paths}
    rules["authors (SpikES)"] = AUTHORS
    res = {}
    for name, p in rules.items():
        res[name] = []
        for er in RATES:
            cfg = NetConfig(**PRESETS["fast"]["net"], ext_rate_range=(er, er))
            ev = Evaluator(cfg, StabilityTask(), [IE], n_trials=1, conn_seed=0)
            r = ev.run_recorded(p, seed=1, phases=[(30.0, 1, 0, 0, -1)], raster_n=200)
            if r["t_blow"] >= 0:
                rE = np.nan
            else:
                sel = r["spk_t"] > 15.0
                rE = np.sum(r["spk_i"][sel] < cfg.NE) / 200 / 15.0
            res[name].append(rE)
            print(f"{name:28s} ext {er:5.1f} Hz -> r_E {rE:6.2f} Hz  (blow-up t={r['t_blow']:.1f})", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, v in res.items():
        ax.plot(RATES, v, "o-", label=name)
    ax.axhline(10, c="r", ls="--"); ax.axvspan(1, 5, color="grey", alpha=.15, label="training range")
    ax.set_xlabel("external input rate per Poisson source (Hz)"); ax.set_ylabel("settled r_E (Hz)")
    ax.legend(fontsize=7); fig.tight_layout()
    out = Path(paths[0]).parent / "input_sweep.png"; fig.savefig(out, dpi=110); print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
