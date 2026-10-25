"""
Degeneracy + "is the CMA covariance the mean-field manifold?" analysis (goes beyond the paper).

For I->E stability rules, mean-field predicts r_E* = -alpha r_I / (beta + (kappa tau_post + gamma tau_pre) r_I).
Moving theta along directions that leave r_E* unchanged costs nothing -> degenerate manifold.
If the paper's reading of the CMA covariance is right, the direction CMA-ES has *shrunk* the most
(smallest-variance eigenvector of C) should align with grad_theta r_E*, the one direction that changes behaviour.

  python degeneracy.py runs/stability_IE_fast_s0.npz runs/stability_IE_fast_s1.npz
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import NetConfig, StabilityTask, Evaluator, IE, rule_to_theta
from analyze import prepost_window
from meta import PRESETS

AUTHORS = np.array([-2.912, -2.328, -1.411, 0.015, 0.716, 1.081])   # SpikES Saves/2212091405_bg_TIF_IE_6pPol_CMA.npz


def rE_star(p, rI):
    tp, tq, a, b, k, g = rule_to_theta(p)
    return -a * rI / (b + (k * tq + g * tp) * rI)


def grad(f, p, eps=1e-4):
    return np.array([(f(p + eps * e) - f(p - eps * e)) / (2 * eps) for e in np.eye(len(p))])


def main(paths):
    runs = {Path(p).stem: np.load(p, allow_pickle=True) for p in paths}
    rules = {k: v["mean"][-1] for k, v in runs.items()}
    rules["authors (SpikES, 400 gens)"] = AUTHORS

    # measure operating rates of every rule in the same network (fast preset, long run)
    cfg = NetConfig(**PRESETS["fast"]["net"]); task = StabilityTask(**PRESETS["fast"]["stab"])
    ev = Evaluator(cfg, task, [IE], n_trials=1, conn_seed=0)
    meas = {}
    for name, p in rules.items():
        rE, rI = [], []
        for seed in range(3):
            r = ev.run_recorded(p, seed=seed, phases=[(40.0, 1, 0, 0, -1)], raster_n=200)
            sel = r["spk_t"] > 20.0
            rE.append(np.sum(r["spk_i"][sel] < cfg.NE) / 200 / 20.0)
            rI.append(np.sum(r["spk_i"][sel] >= cfg.NE) / 50 / 20.0)
        meas[name] = (np.mean(rE), np.mean(rI))
        print(f"{name:32s} r_E={meas[name][0]:6.2f} Hz  r_I={meas[name][1]:6.2f} Hz  "
              f"mean-field r_E*={rE_star(p, meas[name][1]):6.2f} Hz  theta={np.round(rule_to_theta(p), 4)}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    dts = np.linspace(-0.1, 0.1, 401)
    rIs = np.linspace(1, 40, 200)
    for name, p in rules.items():
        ax[0].plot(dts * 1e3, prepost_window(p, dts), label=name)
        ax[1].plot(rIs, [rE_star(p, x) for x in rIs], label=name)
        ax[1].plot(meas[name][1], meas[name][0], "o", color=ax[1].lines[-1].get_color())
    ax[0].axhline(0, c="grey", lw=.5); ax[0].set_xlabel(r"$t_{post}-t_{pre}$ (ms)"); ax[0].set_ylabel(r"$\Delta w/\eta$")
    ax[0].set_title("same task, different rules (degeneracy)"); ax[0].legend(fontsize=7)
    ax[1].axhline(10, c="r", ls="--"); ax[1].set_ylim(0, 40); ax[1].set_xlabel(r"$r_I$ (Hz)")
    ax[1].set_ylabel(r"mean-field $r_E^*$ (Hz)"); ax[1].set_title("mean-field fixed point (dots: measured)")

    # covariance vs mean-field gradient, along the whole trajectory
    print("\n|cos| between grad r_E* and CMA eigenvectors (sorted small->large variance), per run:")
    for name, d in runs.items():
        cos_small, cos_rand = [], []
        rI = meas[name][1]
        for g in range(len(d["mean"])):
            gr = grad(lambda q: rE_star(q, rI), d["mean"][g]); gr /= np.linalg.norm(gr)
            w, V = np.linalg.eigh(d["C"][g])
            cos_small.append(abs(V[:, 0] @ gr))
        gr = grad(lambda q: rE_star(q, rI), d["mean"][-1]); gr /= np.linalg.norm(gr)
        w, V = np.linalg.eigh(d["C"][-1])
        print(f"  {name}: {np.round(np.abs(V.T @ gr), 2)}   (random-direction baseline ~{np.sqrt(2/np.pi/6):.2f})")
        ax[2].plot(cos_small, label=name)
    ax[2].axhline(np.sqrt(2 / np.pi / 6), c="grey", ls="--", label="random baseline")
    ax[2].set_ylim(0, 1); ax[2].set_xlabel("meta-iteration")
    ax[2].set_title(r"|cos(smallest-variance CMA axis, $\nabla_\theta r_E^*$)|"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    out = Path(paths[0]).parent / "degeneracy.png"
    fig.savefig(out, dpi=110); print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
