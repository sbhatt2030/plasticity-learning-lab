"""
Figures in the spirit of Fig 2/3/4 of the paper for a saved meta-learning run:
  python analyze.py runs/stability_IE_fast_s0.npz [--gen-cov 15] [--test-mult 2]
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import NetConfig, StabilityTask, FamiliarityTask, Evaluator, TYPE_NAMES, rule_to_theta
from meta import PRESETS

PNAMES = [r"$\tau_{pre}$", r"$\tau_{post}$", r"$\alpha$", r"$\beta$", r"$\kappa$", r"$\gamma$"]


def prepost_window(rule, dts):
    """Weight change for ONE pre and ONE post spike separated by dt = t_post - t_pre."""
    tp, tq, a, b, k, g = rule_to_theta(rule)
    dw = np.full_like(dts, a + b)
    pos = dts > 0
    dw[pos] += g * np.exp(-dts[pos] / tp)       # post after pre: post spike reads pre trace
    dw[~pos] += k * np.exp(dts[~pos] / tq)      # pre after post: pre spike reads post trace
    return dw


def mean_field(rule, r_pre, r_post):
    """Mean drift of w (Hz units): alpha r_pre + beta r_post + (kappa tau_post + gamma tau_pre) r_pre r_post."""
    tp, tq, a, b, k, g = rule_to_theta(rule)
    return a * r_pre + b * r_post + (k * tq + g * tp) * r_pre * r_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gen-cov", type=int, default=15)
    ap.add_argument("--test-mult", type=float, default=2.0, help="test for this x training duration")
    a = ap.parse_args()
    d = np.load(a.run, allow_pickle=True)
    args = json.loads(str(d["args"]))
    types = list(d["types"])
    mean, C, mloss, ploss = d["mean"], d["C"], d["mean_loss"], d["pop_loss"]
    G = len(mean); nr = len(types)
    out = Path(a.run).with_suffix("")
    out.mkdir(exist_ok=True)

    # ---------- 1. meta-optimisation trajectory + covariance + pre-post window
    fig, ax = plt.subplots(nr + 1, 3, figsize=(15, 3.6 * (nr + 1)), squeeze=False)
    for k, ty in enumerate(types):
        P = mean[:, 6 * k: 6 * k + 6]
        axp = ax[k, 0]
        axp.plot(np.exp(P[:, 0]) * 1e3, label=PNAMES[0]); axp.plot(np.exp(P[:, 1]) * 1e3, label=PNAMES[1])
        axp.set_ylabel("ms"); axp.set_title(f"{ty}: time constants")
        axt = axp.twinx()
        for i in range(2, 6):
            axt.plot(P[:, i], ls="--", label=PNAMES[i], color=f"C{i}")
        axp.legend(loc="upper left", fontsize=8); axt.legend(loc="upper right", fontsize=8)
        axp.set_xlabel("meta-iteration")
        dts = np.linspace(-0.1, 0.1, 401)
        ax[k, 1].plot(dts * 1e3, prepost_window(P[-1], dts), "k")
        ax[k, 1].axhline(0, color="grey", lw=.5); ax[k, 1].axvline(0, color="grey", lw=.5)
        ax[k, 1].set_title(f"{ty}: pre-post protocol (final rule)"); ax[k, 1].set_xlabel(r"$t_{post}-t_{pre}$ (ms)")
        ax[k, 1].set_ylabel(r"$\Delta w / \eta$")
    ax[-1, 0].semilogy(mloss, "k", label="loss(mean rule)"); ax[-1, 0].semilogy(ploss.min(1), ".", label="best in pop")
    ax[-1, 0].set_xlabel("meta-iteration"); ax[-1, 0].legend(); ax[-1, 0].set_title("loss")
    labels = [f"{t}:{p}" for t in types for p in PNAMES]
    for axc, gi, ttl in [(ax[-1, 1], min(a.gen_cov, G - 1), "corr at gen %d" % min(a.gen_cov, G - 1)),
                         (ax[-1, 2], G - 1, "corr at last gen")]:
        c = C[gi]; s = np.sqrt(np.diag(c)); R = c / np.outer(s, s)
        im = axc.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
        axc.set_xticks(range(len(labels))); axc.set_xticklabels(labels, rotation=90, fontsize=7)
        axc.set_yticks(range(len(labels))); axc.set_yticklabels(labels, fontsize=7)
        axc.set_title(ttl); plt.colorbar(im, ax=axc, fraction=.046)
    for k in range(nr):
        ax[k, 2].axis("off")
        ax[k, 2].text(0, .5, f"{types[k]} final rule\n" + "\n".join(
            f"{n}: {v:.4g}" for n, v in zip(["tau_pre (ms)", "tau_post (ms)", "alpha", "beta", "kappa", "gamma"],
                                             rule_to_theta(mean[-1, 6 * k:6 * k + 6]) * np.array([1e3, 1e3, 1, 1, 1, 1]))),
            family="monospace", va="center")
    fig.tight_layout(); fig.savefig(out / "meta_optimisation.png", dpi=110); plt.close(fig)

    # ---------- 2. simulate the final rule for longer than training
    pre = PRESETS[args["preset"]]
    cfg = NetConfig(**pre["net"])
    task = StabilityTask(**pre["stab"]) if args["task"] == "stability" else FamiliarityTask(**pre["fam"])
    ev = Evaluator(cfg, task, [TYPE_NAMES.index(t) for t in types], n_trials=1, conn_seed=args["seed"])
    rng = np.random.default_rng(123)
    ph = task.phases(rng)
    if args["task"] == "stability":
        T = a.test_mult * (task.t_train + task.t_score_phase)
        ph = [(T, 1, 0, 0, -1)]
    runs = {"meta-learned": mean[-1], "initial (no plasticity)": mean[0] * 0 + np.tile(
        [np.log(0.01), np.log(0.01), 0, 0, 0, 0], nr)}
    fig, ax = plt.subplots(4, 2, figsize=(14, 12), gridspec_kw=dict(height_ratios=[2, 1, 1.2, 1.2]))
    mf_lines = []
    for col, (lab, p) in enumerate(runs.items()):
        r = ev.run_recorded(p, seed=7, phases=ph, raster_n=200)
        t = np.arange(len(r["rate"])) * r["rec_dt"]
        ax[0, col].plot(r["spk_t"], r["spk_i"], ",k"); ax[0, col].set_title(f"{lab}: raster (200 E top, 50 I)")
        ax[0, col].set_xlim(0, t[-1] if len(t) else 1)
        ax[1, col].plot(t, r["rate"]); ax[1, col].set_ylabel("E rate (Hz, 1 s filter)")
        if args["task"] == "stability":
            ax[1, col].axhline(task.target, color="r", ls="--")
        tb = 0.0
        for dur, pl, sc, lb, pt in r["phases"]:
            if sc:
                ax[1, col].axvspan(tb, tb + dur, color="C2" if lb == 1 else ("C3" if lb == 2 else "C1"), alpha=.2)
            tb += dur
        ax[1, col].set_xlabel("time (s)")
        for k, ty in enumerate(types):
            m = r["stype"] == TYPE_NAMES.index(ty)
            ax[2 + k if k < 2 else 3, col].hist([r["W0"][m], r["W"][m]], bins=60, label=["initial", "final"],
                                                 histtype="step")
            ax[2 + k if k < 2 else 3, col].set_title(f"{ty} weights (wmax={r['wmax']:.2f})")
            ax[2 + k if k < 2 else 3, col].legend()
        # measured rates over last 20% vs mean-field
        tend = r["spk_t"].max() if len(r["spk_t"]) else 0
        sel = r["spk_t"] > 0.8 * tend
        rE = np.sum(r["spk_i"][sel] < cfg.NE) / 200 / max(0.2 * tend, 1e-9)
        rI = np.sum(r["spk_i"][sel] >= cfg.NE) / 50 / max(0.2 * tend, 1e-9)
        mf_lines.append(f"{lab}: blow-up t={r['t_blow']:.2f} (-1 = none), r_E={rE:.2f} Hz, r_I={rI:.2f} Hz")
        if lab == "meta-learned":
            for k, ty in enumerate(types):
                rp, rq = {"E": rE, "I": rI}[ty[0]], {"E": rE, "I": rI}[ty[1]]
                tp, tq, al, be, ka, ga = rule_to_theta(p[6*k:6*k+6]); c = ka * tq + ga * tp
                if ty[0] != ty[1]:   # IE / EI: fixed point for the postsynaptic rate given the presyn rate
                    pred = -al * rp / (be + c * rp)
                    mf_lines.append(f"   mean-field (eq 13/15) predicted r_{ty[1]} = {pred:.2f} Hz given measured r_{ty[0]}")
                else:                # EE / II: fixes the rate of that population
                    mf_lines.append(f"   mean-field (eq 14/16) predicted r_{ty[0]} = {-(al + be) / c:.2f} Hz")
    if len(types) < 2:
        ax[3, 0].axis("off"); ax[3, 1].axis("off")
    fig.suptitle("\n".join(mf_lines), fontsize=9, family="monospace")
    fig.tight_layout(); fig.savefig(out / "network_activity.png", dpi=110); plt.close(fig)
    print("\n".join(mf_lines)); print("figures in", out)


if __name__ == "__main__":
    main()
