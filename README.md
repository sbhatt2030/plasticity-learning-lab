# ISTA paper review and thought process

A small, exploratory companion to Confavreux et al. (2025), *Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks*. This was deliberately built quickly for learning and experimentation—not as a production simulator or an exact reproduction of the authors' software.

The reference paper is included as [`2025_PLoSCompBio_Confavreux.pdf`](2025_PLoSCompBio_Confavreux.pdf). The supplementary image is [`pcbi.1012910.s001.png`](pcbi.1012910.s001.png).

## What's here

| Directory | Purpose | Best starting point |
| --- | --- | --- |
| [`neuron_lab/`](neuron_lab/) | Interactive local web visualisations: a single conductance-based LIF neuron, a small E/I network, input layer, plasticity rule, and plastic network. | [`neuron_lab/README.md`](neuron_lab/README.md) |
| [`spikes_mini/`](spikes_mini/) | Laptop-scale, pure-Python/Numba recreation of the paper's recurrent spiking-network and CMA-ES plasticity-rule search. Includes saved runs and analysis figures. | [`spikes_mini/README.md`](spikes_mini/README.md) |

## Quick start

Use Python 3.12 or a reasonably recent Python 3 release. The interactive lab uses only the standard library; the meta-learning recreation needs NumPy, Numba, Matplotlib, and `cma`.

```powershell
# Optional but recommended: work in a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install numpy numba matplotlib cma
```

### Explore the visual lab

```powershell
cd neuron_lab
python app.py
```

Visit <http://127.0.0.1:8000>. The home page starts with one neuron; the navigation exposes the network, input-layer, plasticity, and plastic-network views. Stop the server with `Ctrl+C`.

### Inspect or rerun the compact reproduction

```powershell
cd spikes_mini

# Check the authors' reported I→E rule against a no-plasticity baseline.
python check_sim.py

# Run a short stability-rule search (the first run compiles Numba kernels).
python meta.py --task stability --types IE --preset fast --gens 40

# Plot diagnostics for a saved/new run.
python analyze.py runs/stability_IE_fast_s0.npz
```

Saved example outputs live under [`spikes_mini/runs/`](spikes_mini/runs/). See the subproject README for the full experiment list, the paper-to-code map, results, and known deviations.

## Important scope notes

- Network sizes and task durations are reduced for laptop-scale runs; the `fast` preset is intended for iteration, not direct quantitative comparison with the paper.
- The simulator implements the six-parameter “small polynomial” plasticity rule, rather than every model/search space in the source project.
- Results are stochastic. Use fixed seeds when comparing changes, and treat included plots as learning evidence rather than validated scientific conclusions.
- There is no packaging, automated test suite, or locked dependency file yet. If this becomes a durable project, the most useful next cleanup would be a `pyproject.toml`/requirements file and a few smoke tests around the simulation APIs.

## Suggested learning path

1. Start the neuron lab and change one input or parameter at a time.
2. Read the corresponding model comments in [`neuron_lab/app.py`](neuron_lab/app.py).
3. Run `spikes_mini/check_sim.py`, then browse its README’s explanation of the rule and loss.
4. Compare saved runs with `analyze.py`, then explore degeneracy/generalisation using `degeneracy.py` and `input_sweep.py`.

## Provenance

This repository is an independent educational implementation informed by the cited paper and its authors' SpikES repository. It is not the official implementation; consult the paper and original code for research use or exact reproduction.
