"""First visual lab for the conductance-based LIF neuron used in Confavreux et al. (2025).

Run from this folder with:  python app.py
Then open http://127.0.0.1:8000 in a browser.
"""
from __future__ import annotations

import json
import math
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from network import simulate_network
from input_layer import simulate_inputs
from plasticity import simulate_rule
from plastic_network import simulate_plastic_network


ROOT = Path(__file__).parent


def number(source: dict, key: str, default: float) -> float:
    try:
        return float(source.get(key, default))
    except (TypeError, ValueError):
        return default


def events(source: dict, key: str) -> list[float]:
    """Accept a comma/space-separated list of event times in milliseconds."""
    raw = str(source.get(key, ""))
    result = []
    for item in raw.replace(",", " ").split():
        try:
            t = float(item)
            if t >= 0:
                result.append(t * 1e-3)
        except ValueError:
            pass
    return sorted(result)


def simulate(source: dict) -> dict:
    # Paper defaults, all in SI units internally.
    duration = max(1e-3, number(source, "duration_ms", 150.0) * 1e-3)
    dt = max(1e-5, min(1e-3, number(source, "dt_ms", 0.1) * 1e-3))
    tau_m = max(1e-4, number(source, "tau_m_ms", 20.0) * 1e-3)
    tau_a = max(1e-4, number(source, "tau_ampa_ms", 5.0) * 1e-3)
    tau_g = max(1e-4, number(source, "tau_gaba_ms", 10.0) * 1e-3)
    refractory = max(0.0, number(source, "refractory_ms", 5.0) * 1e-3)
    v_rest = number(source, "v_rest_mv", -60.0) * 1e-3
    v_reset = number(source, "v_reset_mv", -60.0) * 1e-3
    v_threshold = number(source, "v_threshold_mv", -50.0) * 1e-3
    e_ampa = number(source, "e_ampa_mv", 0.0) * 1e-3
    e_gaba = number(source, "e_gaba_mv", -80.0) * 1e-3
    w_exc = max(0.0, number(source, "w_exc", 0.12))
    w_inh = max(0.0, number(source, "w_inh", 0.20))
    exc_events, inh_events = events(source, "exc_events_ms"), events(source, "inh_events_ms")

    n_steps = int(math.ceil(duration / dt)) + 1
    exc_i = inh_i = 0
    v, g_a, g_g, refr = v_rest, 0.0, 0.0, 0.0
    trace = {"t_ms": [], "v_mv": [], "g_ampa": [], "g_gaba": [], "spikes_ms": [],
             "exc_events_ms": [x * 1e3 for x in exc_events], "inh_events_ms": [x * 1e3 for x in inh_events]}
    decay_a, decay_g = math.exp(-dt / tau_a), math.exp(-dt / tau_g)
    eps = dt / 2

    for step in range(n_steps):
        t = step * dt
        # A presynaptic event instantly increases its conductance.
        while exc_i < len(exc_events) and abs(exc_events[exc_i] - t) <= eps:
            g_a += w_exc
            exc_i += 1
        while inh_i < len(inh_events) and abs(inh_events[inh_i] - t) <= eps:
            g_g += w_inh
            inh_i += 1

        if refr > 0:
            refr = max(0.0, refr - dt)
            v = v_reset
        else:
            # Forward-Euler implementation of the paper's Eq. 4.
            dv = dt / tau_m * (-(v - v_rest) - g_a * (v - e_ampa) - g_g * (v - e_gaba))
            v += dv
            if v >= v_threshold:
                trace["spikes_ms"].append(t * 1e3)
                v = v_reset
                refr = refractory

        trace["t_ms"].append(round(t * 1e3, 5))
        trace["v_mv"].append(round(v * 1e3, 5))
        trace["g_ampa"].append(round(g_a, 6))
        trace["g_gaba"].append(round(g_g, 6))
        g_a *= decay_a
        g_g *= decay_g

    trace["parameters"] = {
        "duration_ms": duration * 1e3, "dt_ms": dt * 1e3, "tau_m_ms": tau_m * 1e3,
        "tau_ampa_ms": tau_a * 1e3, "tau_gaba_ms": tau_g * 1e3,
        "v_rest_mv": v_rest * 1e3, "v_threshold_mv": v_threshold * 1e3,
        "v_reset_mv": v_reset * 1e3, "refractory_ms": refractory * 1e3,
    }
    return trace


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/api/simulate", "/api/network", "/api/inputs", "/api/plasticity", "/api/plastic-network"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            source = payload if isinstance(payload, dict) else {}
            result = simulate_plastic_network(source) if route == "/api/plastic-network" else (simulate_rule(source) if route == "/api/plasticity" else (simulate_inputs(source) if route == "/api/inputs" else (simulate_network(source) if route == "/api/network" else simulate(source))))
            body = json.dumps(result).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (json.JSONDecodeError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Neuron Lab: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
