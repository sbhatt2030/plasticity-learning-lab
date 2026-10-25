"""
Outer loop: CMA-ES over plasticity-rule parameters (Confavreux et al. 2025, Fig 1 / eq 17).

  python meta.py --task stability --types IE --preset fast --gens 30
  python meta.py --task familiarity --types EE IE --preset paper

Saves runs/<name>.npz with mean/sigma/covariance/loss history (analyse with analyze.py).
"""
import argparse, time, json
from pathlib import Path
import numpy as np
import cma
from sim import NetConfig, StabilityTask, FamiliarityTask, Evaluator, TYPE_NAMES

PRESETS = {
    # same in-degree as the paper everywhere; "fast" shrinks N and protocol durations
    "paper": dict(net=dict(NE=2000, NI=500), stab=dict(), fam=dict()),
    "fast": dict(net=dict(NE=1000, NI=250),
                 stab=dict(t_train=10.0, t_score_phase=10.0, t_window=5.0),
                 fam=dict(t_train=(10.0, 12.0), t_score=3.0)),
}
NULL_RULE = [np.log(0.01), np.log(0.01), 0, 0, 0, 0]   # initial rule in the paper: no plasticity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["stability", "familiarity"], default="stability")
    ap.add_argument("--types", nargs="+", default=["IE"], help="plastic synapse types, e.g. IE or EE IE")
    ap.add_argument("--preset", choices=list(PRESETS), default="fast")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--trials", type=int, default=4, help="networks per rule (N_trials)")
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--reg", type=float, default=0.0, help="L1 penalty on rule parameters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    pre = PRESETS[a.preset]
    cfg = NetConfig(**pre["net"])
    task = StabilityTask(**pre["stab"]) if a.task == "stability" else FamiliarityTask(**pre["fam"])
    types = [TYPE_NAMES.index(t) for t in a.types]
    ev = Evaluator(cfg, task, types, n_trials=a.trials, conn_seed=a.seed)

    x0 = np.tile(NULL_RULE, len(types)).astype(float)
    es = cma.CMAEvolutionStrategy(x0, a.sigma0, {"popsize": a.popsize, "seed": a.seed + 1, "verbose": -9})
    name = a.name or f"{a.task}_{''.join(a.types)}_{a.preset}_s{a.seed}"
    out = Path(__file__).parent / "runs"; out.mkdir(exist_ok=True)
    H = dict(mean=[], sigma=[], C=[], pop=[], pop_loss=[], mean_loss=[])

    for g in range(a.gens):
        t0 = time.time()
        X = es.ask()
        # like Outerloop_CMA_ES: also score the current mean (for plotting only)
        L, _ = ev.score(X + [es.mean.copy()], seed=a.seed * 100000 + g)
        pop_loss, mean_loss = L[:-1], L[-1]
        es.tell(X, list(pop_loss + a.reg * np.abs(np.array(X)).sum(1)))
        H["mean"].append(es.mean.copy()); H["sigma"].append(es.sigma); H["C"].append(np.array(es.C))
        H["pop"].append(np.array(X)); H["pop_loss"].append(pop_loss); H["mean_loss"].append(mean_loss)
        print(f"gen {g:3d}  loss(mean rule) {mean_loss:9.4f}  best {pop_loss.min():9.4f}  "
              f"sigma {es.sigma:.3f}  ({time.time() - t0:.0f}s)  mean {np.round(es.mean, 3)}", flush=True)
        np.savez(out / f"{name}.npz", **{k: np.array(v) for k, v in H.items()},
                 types=np.array(a.types), args=json.dumps(vars(a)))
    print("saved", out / f"{name}.npz")


if __name__ == "__main__":
    main()
