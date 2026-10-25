# Neuron Lab — Step 1

An interactive, local HTML visualizer for one conductance-based leaky integrate-and-fire neuron. It implements the first neuron model in Confavreux et al. (2025), Eq. 4–5, with the paper defaults.

## Start it

From this directory:

```powershell
python app.py
```

Open <http://127.0.0.1:8000>. Stop it with `Ctrl+C` in the terminal.

## Experiments to try

1. Clear all events: the voltage stays at its resting value.
2. Use one excitatory event: its effect fades before it can usually make the neuron spike.
3. Cluster several excitatory events: their AMPA conductances add and can cross threshold.
4. Place inhibitory events immediately before the excitatory cluster: GABA conductance counters it.
5. Raise the threshold or lower the excitatory weight: spiking becomes harder.

The next step is a small fixed excitatory/inhibitory network, retaining these same neuron and synapse views.

## Step 2.1

Open <http://127.0.0.1:8000/network.html> while the server is running. This page uses eight excitatory and two inhibitory neurons with seeded random wiring. It applies one controlled external excitatory pulse to a chosen neuron, then subtracts the activity of an otherwise identical baseline run. The green response trace is therefore the network-wide effect caused by that one input rather than by different random noise.

Keep the activity and connectivity seeds fixed while changing exactly one parameter. The connection probability changes how many directed links are available for an input to spread through; the connectivity seed changes *which* links exist.
