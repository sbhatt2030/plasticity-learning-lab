# spikes_mini — a small-scale recreation of Confavreux et al. 2025 (PLoS Comp Biol)

*"Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks"*
Authors' code: https://github.com/VogelsLab/SpikES (C++/Auryn + Python). This is a pure-Python/numba
rewrite of the core pipeline, built from the paper plus the authors' repo (loss constants, protocol
timings, rule update order were read from their source).

## 1. What the paper actually does

```
          outer loop (CMA-ES)                               inner loop (spiking net)
  N(m, sigma^2 C)  --sample lambda rules theta_k-->  for each rule, N_trials networks:
        ^                                              random input rate, init IE weight, w_max
        |                                              simulate 30-50 s with plasticity ON
        +----- rank rules by mean loss <---------      loss from the E-population rate
```

**Rule ("small polynomial", eq 9)**, per synapse i→j, 6 params θ = [τ_pre, τ_post, α, β, κ, γ]:

| event | update |
|---|---|
| presynaptic spike | `w += η (α + κ · y_post)` |
| postsynaptic spike | `w += η (β + γ · x_pre)` |

α and β are **non-Hebbian** (they fire on one side's spike alone). γ and κ are the **Hebbian**
pair terms (the classic STDP window). τ values are optimised in log space. η = 0.01, and w is clipped to [0, w_max].

**Stability loss** (`CheckerScorer_Stability.cpp`): the E rate is low-passed with τ = 1 s, and the loss is
`<(r − 10)² / (r + 2)>` over a random 5 s window after 30–45 s of plastic dynamics. If r leaves
[0.1, 100] Hz the network has "blown up" and the loss is `160·(e^{−t_blow/T} + 1)`, which is 160–320, so an earlier blow-up gets a worse loss.

**Familiarity loss** (S1 Fig, `CheckerScorer_FamDet_Lbasic.cpp`): train on a familiar input pattern, then
freeze plasticity and present familiar and novel patterns. The loss is hinged: familiar is penalised if r < 20 Hz
(`1 − r/20`), novel is penalised if r > 10 Hz (`(r−10)/90`).

**Mean-field (eq 13–16).** Average the rule over Poisson spiking. The drift of w is then
`α r_pre + β r_post + (κ τ_post + γ τ_pre) r_pre r_post`. Setting it to zero gives the steady-state rates:

* **I→E:** `r_E* = −α r_I / (β + (κτ_post + γτ_pre) r_I)`. The rule sets the E rate directly, which is why it works best.
* **E→I:** fixes `r_I` as a function of `r_E`. It controls E only indirectly, through the E→I→E loop.
* **E→E:** `r_E* = −(α+β)/(κτ_post + γτ_pre)`. **The paper prints `r_inh*` here, which looks like a typo.**
* **I→I:** `r_I* = −(α+β)/(κτ_post + γτ_pre)`. This pins only the inhibitory rate. Nothing forces r_E to 10 Hz, so it
  makes sense that no I→I stability rule was found.

The fixed point depends on (α, β, κτ_post + γτ_pre) only through ratios. So a **whole manifold of θ gives the
same behaviour**, which is where the **degeneracy** and the **α–β anti-correlation** in the CMA covariance come from:
scaling α and β together keeps the ratio fixed.

## 2. Code map

| file | role | paper / SpikES counterpart |
|---|---|---|
| `sim.py` `_run_one` | event-driven conductance LIF E/I net, spike-triggered plasticity on any of EE/EI/IE/II | Auryn sim, `SixParamConnection.cpp` |
| `sim.py` `StabilityTask`, `FamiliarityTask` | protocols and losses | `sim_innerloop_*.cpp`, `CheckerScorer_*.cpp` |
| `sim.py` `Evaluator` | λ rules × N_trials networks in parallel (numba `prange`) | `Innerloop_*.py` |
| `meta.py` | CMA-ES outer loop (pycma), logs mean/σ/C/losses | `Outerloop_CMA_ES.py` |
| `analyze.py` | parameter trajectories, loss, pre-post window, correlation matrix, long test run, mean-field check | Figs 2–5 |
| `degeneracy.py` | compares rules across seeds and the authors' rule; CMA covariance vs mean-field gradient | Fig 8 + new |
| `input_sweep.py` | generalisation of equal-loss rules to unseen input rates | new |
| `fam_control.py` | order-reversal control for familiarity detection | new |
| `check_sim.py` | validation: authors' rule vs null rule | — |

```
python meta.py --task stability   --types IE    --preset fast --gens 40            # Fig 2E / 3C
python meta.py --task stability   --types EE    --preset fast                      # Fig 2C
python meta.py --task familiarity --types IE    --preset fast --gens 30            # Fig 4
python meta.py --task familiarity --types EE IE --preset fast                      # Fig 6D (co-active)
python meta.py ... --seed 1                                                        # degeneracy (Fig 8)
python analyze.py runs/<run>.npz
```

## 3. Deviations from the paper (deliberate, for laptop-scale runs)

* **Size:** 2000E/500I (`paper` preset) or 1000E/250I (`fast`) instead of 8000/2000. **The in-degree is kept the same**
  (160 E + 40 I recurrent inputs and 250 external Poisson inputs per neuron), so the input statistics per neuron match.
* **`fast` preset:** 10 s train + 10 s scoring phase instead of 30 + 20 s. Population λ = 10 (same as their notebook) and
  4 trials per rule (they used 10).
* External input is 250 independent Poisson inputs per neuron, summed exactly as one event-driven process.
* **Familiarity task:** no receptive-field topography (`RFConnection`, radius 8). Each neuron instead gets
  k ~ Bin(250, 0.1) inputs from each of two non-overlapping patterns. Training uses the familiar pattern continuously
  rather than 2 s on / 0.1 s off.
* Only the small polynomial rule is implemented. The big polynomial and MLP rules are not.
* **Validation:** the authors' final I→E stability rule (from `Saves/2212091405_bg_TIF_IE_6pPol_CMA.npz`:
  τ_pre = 54 ms, τ_post = 97 ms, α = −1.41, β = 0.015, κ = 0.72, γ = 1.08) gets loss 0.001–0.013 in this simulator.
  Their log reports 0.012. The no-plasticity rule blows up within 1–2 s (their initial loss is ≈ 194).

## 4. Results of the recreation (fast preset, 40 generations, λ = 10, 4 trials per rule)

Figures are in `runs/<run>/meta_optimisation.png`, `runs/<run>/network_activity.png`, `runs/degeneracy.png` and `runs/input_sweep.png`.

**Stability, I→E (paper Fig 2E / 3C).** Reproduced.
* The no-plasticity rule blows up within ~3 s (E rate → 100 Hz).
* Within ~3 generations, CMA-ES finds rules that hold the E rate near 10 Hz, and they stay stable for 2× the training duration.
* The covariance shows the paper's hallmark **strong α–β (non-Hebbian) anti-correlation** (≈ −0.9).

**Degeneracy (paper Fig 8).** Reproduced. Two seeds from the same null starting rule converged to rules that look very different under the pre-post protocol:

| rule | τ_pre | τ_post | α | β | κ | γ | final training loss |
|---|---|---|---|---|---|---|---|
| seed 0 | 4.8 ms | 10 ms | −1.29 | **+1.19** | **−1.32** | 0.66 | ~0.2–1 |
| seed 1 | 12 ms | 15 ms | −0.14 | ≈0 | 0.74 | 0.67 | ~0.01 |
| authors (400 gens) | 54 ms | 98 ms | −1.41 | ≈0 | 0.72 | 1.08 | 0.012 |

### 4a. New: the CMA covariance *is* the mean-field constraint (quantitative check)
The paper reads the CMA-ES covariance only "heuristically". Here is a direct test. At every generation, take the
covariance's **smallest-variance eigenvector** and compare it with ∇_θ r_E*, the gradient of the mean-field fixed point (eq 13).

* In both runs, |cos| ≈ 0.99 from generation ~3 onwards. The random-direction baseline is 0.33.
* Every other eigenvector has |cos| ≤ 0.1.

So CMA-ES has shrunk exactly the one direction that changes the network's steady state, and stays broad over the remaining
5-D degenerate manifold. This turns the paper's heuristic into a measurable statement.

### 4b. New: rules that are degenerate on the task are *not* degenerate outside it
The mean-field curve r_E*(r_I) = −α r_I / (β + c r_I), with c = κτ_post + γτ_pre, has two regimes:
* **β ≈ 0 (seed 1, authors' rule):** r_E* ≈ −α/c, a flat line. This is an **absolute** E-rate set-point, independent of inhibition and input.
* **β dominant (seed 0):** r_E* ≈ (−α/β)·r_I, a steep line. This is an **E/I ratio** controller. It hits 10 Hz only when the input happens to put r_I near 9 Hz.

The prediction is that seed 0 will fail when the input changes. To test it, sweep the external Poisson rate (training range 1–5 Hz), 30 s runs:

| ext rate (Hz) | 1 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|
| seed 0 r_E | 1.6 | 9.0 | 14.7 | 22.4 | **32.1** |
| seed 1 r_E | 3.2 | 9.7 | 9.5 | 9.2 | 8.9 |
| authors r_E | 7.4 | 10.0 | 10.0 | 9.9 | 9.9 |

All three rules have low training loss, but the loss cannot tell them apart and their generalisation differs a lot.
Mean-field theory predicted which one would fail before any test simulation was run. This connects to the paper's point
that the loss alone doesn't constrain plausibility, and it gives a *cheap, analytic* way to rank degenerate solutions.

### 4c. Familiarity detection with a single I→E rule (paper Fig 4, "IE alone suffices")
Reproduced qualitatively. The loss fell from 22 to ~0.04 in 30 generations. The final rule:

| τ_pre | τ_post | α | β | κ | γ |
|---|---|---|---|---|---|
| 6.7 ms | 18.7 ms | −2.12 | 0.71 | 0.26 | 0.58 |

Pre-post window: all pairs depress, with a potentiating tuft.

**Order-reversal control** (`fam_control.py`, raw spike counts, 3 networks × 2 presentation orders):

| | familiar | novel |
|---|---|---|
| rate range | 25–53 Hz | 6.5–20 Hz |

The effect held in every seed and both orders, so it isn't an artefact of slow rate drift.

Caveats, and why they matter:
* **Implausible rates.** During training the network sits at 40–60 Hz and familiar responses reach 50 Hz. The loss only
  constrains the test windows, so nothing penalises this. This is exactly the plausibility problem the paper raises for
  its larger search spaces, and it already shows up here.
* **No plastic-free baseline.** The paper's S5 control (no plasticity) is impossible in this network, which blows up without plasticity.
* **Open mechanistic question.** Mean-field puts the per-synapse fixed point at r_E ≈ 38 Hz, which is a *homeostatic* force.
  How a homeostatic-looking I→E rule produces pattern-*specific* disinhibition isn't obvious. A good first analysis is to
  split I→E weights by the postsynaptic neuron's familiar-pattern drive k_fam.

## 5. Extension ideas (grounded in what the recreation showed)

1. **Map the degenerate manifold instead of sampling one point of it.** ES returns one rule per run (the paper's stated limitation).
   * Run many cheap restarts, or use quality-diversity search (MAP-Elites, with descriptors such as the "β/c regime" or the
     pre-post window shape), or simulation-based inference: fit a posterior over θ given "loss < ε".
   * 4a suggests a shortcut: sample *along* the flat CMA eigen-directions, or along the analytic iso-r_E* surface, and only
     simulate to confirm.
2. **Use the robustness gap as a selection criterion.** 4b shows that equal-loss rules differ in how they generalise.
   * Add input-rate / network-size / connectivity perturbations to the *training* distribution (domain randomisation),
     or add an explicit term that penalises the slope ∂r_E*/∂r_I.
   * A more interesting question is whether biology picks the "absolute set-point" family. The prediction is that a
     single-spike (non-Hebbian) postsynaptic term β should be ≈ 0 at I→E synapses. That is experimentally testable.
3. **Mean-field–informed search spaces.** Reparameterise θ as (r_E*, stability margin, nuisance directions) so that the
   optimiser works in behaviourally meaningful coordinates and the degeneracy is factored out explicitly.
4. **Replace the covariance heuristic with principled sensitivity analysis.** Options: finite-difference Hessian of the
   loss at convergence, active subspaces, or Fisher information. Check against CMA's C (as in 4a) for the larger rule
   spaces (big polynomial / MLP), where the paper found C "much sparser" but could not interpret it.
5. **Correlation corrections to the mean-field.** Measured rates differ from the rate-based prediction by 10–20%
   (e.g. seed 0: 11.6 Hz measured vs 14.1 Hz predicted). Short τ (5–10 ms) rules pick up spike-correlation terms, and
   weights pile up at the w = 0 boundary. Adding a pairwise correlation term (as in Kempter/Gerstner STDP theory) would
   tell you when "equivalent" rules diverge.
6. **The loss-design bottleneck (the paper's main negative result).** The big-polynomial and MLP rules reached low loss with
   implausible dynamics. Possible fixes:
   * multi-objective ES (Pareto front of task loss vs plausibility statistics: CV of ISI, Fano factor, weight bimodality);
   * a learned plausibility critic;
   * an SBI "filter" approach as in the authors' follow-up (ref [49]).
   An incremental version is easy here: add CV_ISI / synchrony penalties to `_run_one` and see whether the degenerate set shrinks.
7. **Why I→I fails, and co-active rules.** Eq 16 shows that an isolated I→I rule can only pin r_I, so it cannot target the E
   rate. A natural extension is a co-active II+EI or II+IE search, checking whether the mean-field conditions compose
   (the joint fixed point of two rules). Test it with `--types EI II`.
8. **Engineering.** A GPU (JAX) batched simulator would allow λ ≈ 100 and 10+ trials. That's needed for the 21-parameter
   polynomial and to tame the noisy loss seen here (the mean-rule loss fluctuates 0.2–5 between generations with only 4 trials).
