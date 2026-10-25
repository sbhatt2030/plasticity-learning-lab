"""
Inner loop: batched, event-driven simulation of recurrent E/I conductance-based LIF
networks with spike-triggered "small polynomial" plasticity (Confavreux et al. 2025, eq. 9):

    dw_ij/dt = eta * [ alpha*S_i + beta*S_j + gamma*S_j*x_i + kappa*S_i*y_j ]

  i = presynaptic, j = postsynaptic, x_i / y_j = pre / post spike traces (tau_pre, tau_post).
  On a presynaptic spike : w += eta*(alpha + kappa*y_j)
  On a postsynaptic spike: w += eta*(beta  + gamma*x_i)
  (identical to SixParamConnection.cpp in github.com/VogelsLab/SpikES)

Every synapse type (EE, EI, IE, II) can carry its own rule, so single-active and co-active
settings are both supported. Each batch element is an independent network (one rule x one
trial), run in parallel with numba.prange -- the equivalent of the paper's CMA-ES population.

Network: the Vogels, Sprekeler et al. 2011 network used for the stability task (Methods, eq 4-5),
scaled down in size but with the same in-degree per neuron (see NetConfig).
"""
import numpy as np
from numba import njit, prange
from dataclasses import dataclass

EE, EI, IE, II = 0, 1, 2, 3            # synapse type index: <pre><post>
TYPE_NAMES = ["EE", "EI", "IE", "II"]


@dataclass
class NetConfig:
    NE: int = 2000
    NI: int = 500
    # paper: 10000 neurons at 2% -> 160 E + 40 I inputs per neuron. Keep in-degree when scaling.
    k_in_E: int = 160
    k_in_I: int = 40
    w_init: tuple = (0.03, 0.03, None, 0.3)   # EE, EI, IE (sampled per trial), II
    wie_range: tuple = (0.0, 0.3)
    wmax_range: tuple = (2.0, 3.0)            # log-uniform per trial, for plastic synapses
    eta: float = 0.01
    # external Poisson drive: 5000 inputs at 5% -> 250 inputs per neuron, weight 0.1
    n_ext: int = 250
    w_ext: float = 0.1
    ext_rate_range: tuple = (1.0, 5.0)
    dt: float = 1e-4
    # rate checker (CheckerScorer_Stability): population rate low-passed with tau 1 s
    tau_checker: float = 1.0
    rate_min: float = 0.1
    rate_max: float = 100.0


def build_connectivity(cfg: NetConfig, seed: int):
    """Fixed-in-degree random connectivity. Returns CSR by presyn and CSR-by-post index lists."""
    rng = np.random.default_rng(seed)
    NE, NI = cfg.NE, cfg.NI
    N = NE + NI
    pre_l, post_l = [], []
    for j in range(N):
        pe = rng.choice(NE, cfg.k_in_E, replace=False)
        pi = NE + rng.choice(NI, cfg.k_in_I, replace=False)
        pre_l.append(np.concatenate([pe, pi]))
        post_l.append(np.full(cfg.k_in_E + cfg.k_in_I, j))
    pre = np.concatenate(pre_l).astype(np.int32)
    post = np.concatenate(post_l).astype(np.int32)
    stype = ((pre >= NE).astype(np.int32) * 2 + (post >= NE).astype(np.int32)).astype(np.int32)
    # sort by presyn -> forward CSR
    order = np.argsort(pre, kind="stable")
    pre, post, stype = pre[order], post[order], stype[order]
    out_ptr = np.zeros(N + 1, np.int64)
    np.add.at(out_ptr, pre + 1, 1)
    out_ptr = np.cumsum(out_ptr)
    # backward CSR (post -> synapse ids)
    border = np.argsort(post, kind="stable").astype(np.int64)
    in_ptr = np.zeros(N + 1, np.int64)
    np.add.at(in_ptr, post + 1, 1)
    in_ptr = np.cumsum(in_ptr)
    return dict(pre=pre, post=post, stype=stype, out_ptr=out_ptr, in_ptr=in_ptr, in_syn=border)


# --------------------------------------------------------------------------------------------
# core simulator (one network); phases describe the protocol
#   phase arrays (length P): duration [s], plastic (0/1), scored (0/1), label (0 none,1 fam,2 nov),
#   input pattern index (-1 = background only)
# --------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _run_one(seed, NE, NI, pre, post, stype, out_ptr, in_ptr, in_syn,
             w, wmax, theta, active, eta,
             ext_rate, n_ext, w_ext, pat_k, pat_rate,
             ph_dur, ph_plastic, ph_scored, ph_label, ph_pattern,
             dt, tau_checker, rate_min, rate_max,
             task, target, r_cut_fam, r_cut_nov,
             rec_every, rec_rate, rec_raster_n, rec_spk_t, rec_spk_i):
    np.random.seed(seed)
    N = NE + NI
    tau_m, Vrest, Vth, Vreset, tref = 20e-3, -60e-3, -50e-3, -60e-3, 5e-3
    EA, EG, tA, tG = 0.0, -80e-3, 5e-3, 10e-3
    dA, dG = np.exp(-dt / tA), np.exp(-dt / tG)

    V = Vrest + (Vth - Vrest) * np.random.random(N)
    gA = np.zeros(N); gG = np.zeros(N)
    refr = np.zeros(N)
    # traces per synapse type: x[t, i] (pre, tau_pre_t), y[t, j] (post, tau_post_t)
    x = np.zeros((4, N)); y = np.zeros((4, N))
    dx = np.empty(4); dy = np.empty(4)
    for t in range(4):
        dx[t] = np.exp(-dt / theta[t, 0]); dy[t] = np.exp(-dt / theta[t, 1])
    spk = np.zeros(N, np.bool_)

    # external input: superposition of n_ext Poisson inputs per neuron, event-driven
    next_ext = np.zeros(N)
    cur_rate = np.zeros(N)

    rate = 0.5 * (rate_min + rate_max)
    drate = np.exp(-dt / tau_checker)
    loss_acc = 0.0; n_scored = 0
    t_now = 0.0; t_blow = -1.0
    step = 0; n_rec = 0; n_spk_rec = 0
    T_total = 0.0
    for p in range(ph_dur.shape[0]):
        T_total += ph_dur[p]

    for p in range(ph_dur.shape[0]):
        # set input rates for this phase, resample next event times (memoryless => exact)
        pat = ph_pattern[p]
        for n in range(N):
            if pat >= 0:
                k = pat_k[pat, n]
                cur_rate[n] = (n_ext - k) * ext_rate + k * pat_rate
            else:
                cur_rate[n] = n_ext * ext_rate
            next_ext[n] = t_now + np.random.exponential(1.0 / cur_rate[n])
        plastic = ph_plastic[p] == 1
        n_steps = int(round(ph_dur[p] / dt))
        for s in range(n_steps):
            t_now += dt
            nE_spk = 0
            # ---- neurons
            for n in range(N):
                while next_ext[n] <= t_now:
                    gA[n] += w_ext
                    next_ext[n] += np.random.exponential(1.0 / cur_rate[n])
                gA[n] *= dA; gG[n] *= dG
                spk[n] = False
                if refr[n] > 0.0:
                    refr[n] -= dt
                    continue
                v = V[n]
                v += dt / tau_m * (-(v - Vrest) - gA[n] * (v - EA) - gG[n] * (v - EG))
                if v > Vth:
                    v = Vreset; refr[n] = tref; spk[n] = True
                    if n < NE:
                        nE_spk += 1
                V[n] = v
            # ---- propagate + plasticity
            for n in range(N):
                if not spk[n]:
                    continue
                exc = n < NE
                for s_ in range(out_ptr[n], out_ptr[n + 1]):
                    j = post[s_]
                    if exc:
                        gA[j] += w[s_]
                    else:
                        gG[j] += w[s_]
                    ty = stype[s_]
                    if plastic and active[ty]:   # on-pre update: alpha + kappa*y_post
                        nw = w[s_] + eta * (theta[ty, 2] + theta[ty, 5] * y[ty, j])
                        w[s_] = min(max(nw, 0.0), wmax)
                if plastic:
                    for q in range(in_ptr[n], in_ptr[n + 1]):  # on-post update: beta + gamma*x_pre
                        s_ = in_syn[q]
                        ty = stype[s_]
                        if active[ty]:
                            nw = w[s_] + eta * (theta[ty, 3] + theta[ty, 4] * x[ty, pre[s_]])
                            w[s_] = min(max(nw, 0.0), wmax)
            # ---- traces
            for t in range(4):
                if active[t]:
                    for n in range(N):
                        x[t, n] *= dx[t]; y[t, n] *= dy[t]
                        if spk[n]:
                            x[t, n] += 1.0; y[t, n] += 1.0
            # ---- rate checker / loss
            rate = rate * drate + nE_spk / (tau_checker * NE)
            if rec_every > 0 and step % rec_every == 0 and n_rec < rec_rate.shape[0]:
                rec_rate[n_rec] = rate; n_rec += 1
            if rec_raster_n > 0:
                for n in range(N):
                    if spk[n] and (n < rec_raster_n or (n >= NE and n < NE + rec_raster_n // 4)) \
                            and n_spk_rec < rec_spk_t.shape[0]:
                        rec_spk_t[n_spk_rec] = t_now; rec_spk_i[n_spk_rec] = n; n_spk_rec += 1
            step += 1
            if rate < rate_min or rate > rate_max:
                t_blow = t_now
                break
            if ph_scored[p] == 1:
                n_scored += 1
                if task == 0:     # stability: (r - target)^2 / (r + 2)
                    loss_acc += (rate - target) ** 2 / (rate + 2.0)
                else:             # familiarity: hinge losses of S1 Fig
                    if ph_label[p] == 1 and rate <= r_cut_fam:
                        loss_acc += 1.0 - rate / r_cut_fam
                    elif ph_label[p] == 2 and rate >= r_cut_nov:
                        loss_acc += (rate - r_cut_nov) / (100.0 - r_cut_nov)
        if t_blow >= 0.0:
            break

    if t_blow >= 0.0:
        if task == 0:
            loss = 160.0 * (np.exp(-t_blow / T_total) + 1.0)
        else:
            loss = np.exp(-t_blow / T_total) + 42.0
    else:
        loss = loss_acc / max(n_scored, 1)
    return loss, t_blow, n_rec, n_spk_rec


@njit(parallel=True, cache=True)
def _run_batch(seeds, NE, NI, pre, post, stype, out_ptr, in_ptr, in_syn,
               W, wmax, thetas, active, eta, ext_rate, n_ext, w_ext, pat_k, pat_rate,
               ph_dur, ph_plastic, ph_scored, ph_label, ph_pattern,
               dt, tau_checker, rate_min, rate_max, task, target, r_cut_fam, r_cut_nov):
    B = seeds.shape[0]
    losses = np.zeros(B); tblow = np.zeros(B)
    dummy_r = np.zeros(1); dummy_t = np.zeros(1); dummy_i = np.zeros(1, np.int32)
    for b in prange(B):
        l, tb, _, _ = _run_one(seeds[b], NE, NI, pre, post, stype, out_ptr, in_ptr, in_syn,
                               W[b], wmax[b], thetas[b], active, eta,
                               ext_rate[b], n_ext, w_ext, pat_k, pat_rate[b],
                               ph_dur[b], ph_plastic[b], ph_scored[b], ph_label[b], ph_pattern[b],
                               dt, tau_checker, rate_min, rate_max, task, target, r_cut_fam, r_cut_nov,
                               0, dummy_r, 0, dummy_t, dummy_i)
        losses[b] = l; tblow[b] = tb
    return losses, tblow


# --------------------------------------------------------------------------------------------
# rule encoding: one rule = [log tau_pre, log tau_post, alpha, beta, kappa, gamma] (SpikES order)
# --------------------------------------------------------------------------------------------
def rule_to_theta(r):
    """-> row [tau_pre, tau_post, alpha, beta, kappa, gamma] in the simulator's layout."""
    return np.array([np.exp(r[0]), np.exp(r[1]), r[2], r[3], r[4], r[5]])


def thetas_from_params(params, plastic_types):
    th = np.tile(np.array([0.01, 0.01, 0, 0, 0, 0.0]), (4, 1))
    for k, ty in enumerate(plastic_types):
        th[ty] = rule_to_theta(params[6 * k: 6 * k + 6])
    return th


# --------------------------------------------------------------------------------------------
# task protocols
# --------------------------------------------------------------------------------------------
@dataclass
class StabilityTask:
    """Paper / bg_TIF_IE_6pPol.yaml: train 30 s, then a 5 s scoring window drawn at a random
    time within the following 20 s. Loss <(r - 10)^2/(r + 2)> over the window."""
    t_train: float = 30.0
    t_score_phase: float = 20.0
    t_window: float = 5.0
    target: float = 10.0
    task_id: int = 0

    def phases(self, rng):
        lns = rng.uniform(self.t_train, self.t_train + self.t_score_phase - self.t_window)
        # dur, plastic, scored, label, pattern
        return [(lns, 1, 0, 0, -1), (self.t_window, 1, 1, 0, -1)]


@dataclass
class FamiliarityTask:
    """FamDet_Lbasic: train with the familiar pattern(s), break, plasticity frozen,
    then score a familiar and a novel presentation (random order). Loss = S1 Fig hinges.
    Simplification vs paper: no receptive-field topography; each neuron receives k ~ Bin(250, 0.1)
    inputs from each (non-overlapping) pattern, which fire at 15-20 Hz when the pattern is on."""
    t_train: tuple = (30.0, 35.0)
    t_break: tuple = (1.0, 3.0)
    t_score: float = 5.0
    r_cut_fam: float = 20.0
    r_cut_nov: float = 10.0
    frac_active: float = 0.1       # 500 of 5000 inputs
    pat_rate: tuple = (15.0, 20.0)
    task_id: int = 1

    def phases(self, rng):
        lt = rng.uniform(*self.t_train); lb = rng.uniform(*self.t_break)
        fam, nov = (0, 1, 1), (1, 2, 1)   # (pattern, label, _)
        tests = [fam, nov] if rng.random() < 0.5 else [nov, fam]
        return [(lt, 1, 0, 0, 0), (lb, 0, 0, 0, -1),
                (self.t_score, 0, 1, tests[0][1], tests[0][0]), (lb, 0, 0, 0, -1),
                (self.t_score, 0, 1, tests[1][1], tests[1][0])]


class Evaluator:
    """Scores a population of rules: each rule on n_trials networks (random input rate,
    initial IE weight, wmax) -- mean loss per rule, like Innerloop.score()."""

    def __init__(self, cfg: NetConfig, task, plastic_types, n_trials=4, conn_seed=0):
        self.cfg, self.task, self.types, self.n_trials = cfg, task, list(plastic_types), n_trials
        self.conn = build_connectivity(cfg, conn_seed)
        self.active = np.zeros(4, np.bool_)
        self.active[self.types] = True
        N = cfg.NE + cfg.NI
        prng = np.random.default_rng(conn_seed + 1)
        self.pat_k = prng.binomial(cfg.n_ext, getattr(task, "frac_active", 0.0),
                                   size=(2, N)).astype(np.float64)

    def _trial_setup(self, rng):
        c = self.cfg
        w = np.array([c.w_init[0], c.w_init[1], rng.uniform(*c.wie_range), c.w_init[3]])
        W = w[self.conn["stype"]].astype(np.float64)
        wmax = float(np.exp(rng.uniform(*np.log(c.wmax_range))))
        return W, wmax, rng.uniform(*c.ext_rate_range)

    def _pack(self, param_list, rng):
        B = len(param_list) * self.n_trials
        W, wmax, ext, th, seeds, prates, phs = [], [], [], [], [], [], []
        for p in param_list:
            theta = thetas_from_params(p, self.types)
            for _ in range(self.n_trials):
                w_, wm, er = self._trial_setup(rng)
                W.append(w_); wmax.append(wm); ext.append(er); th.append(theta)
                seeds.append(rng.integers(1 << 30))
                prates.append(rng.uniform(*getattr(self.task, "pat_rate", (0, 0))))
                phs.append(self.task.phases(rng))
        P = max(len(p) for p in phs)
        arr = np.zeros((5, B, P))
        arr[4] = -1
        for b, ph in enumerate(phs):
            for k, row in enumerate(ph):
                arr[:, b, k] = row
        return dict(W=np.stack(W), wmax=np.array(wmax), thetas=np.stack(th),
                    seeds=np.array(seeds, np.int64), ext=np.array(ext), prate=np.array(prates),
                    dur=arr[0].copy(), plastic=arr[1].astype(np.int64), scored=arr[2].astype(np.int64),
                    label=arr[3].astype(np.int64), pattern=arr[4].astype(np.int64))

    def score(self, param_list, seed):
        rng = np.random.default_rng(seed)
        d = self._pack(param_list, rng)
        c, t, cn = self.cfg, self.task, self.conn
        losses, tblow = _run_batch(
            d["seeds"], c.NE, c.NI, cn["pre"], cn["post"], cn["stype"], cn["out_ptr"],
            cn["in_ptr"], cn["in_syn"], d["W"], d["wmax"], d["thetas"], self.active, c.eta,
            d["ext"], float(c.n_ext), c.w_ext, self.pat_k, d["prate"],
            d["dur"], d["plastic"], d["scored"], d["label"], d["pattern"],
            c.dt, c.tau_checker, c.rate_min, c.rate_max, t.task_id,
            getattr(t, "target", 10.0), getattr(t, "r_cut_fam", 20.0), getattr(t, "r_cut_nov", 10.0))
        return losses.reshape(len(param_list), self.n_trials).mean(1), tblow

    def run_recorded(self, params, seed, phases=None, raster_n=200, rec_dt=0.01, plastic_override=None):
        """Single network with rate + raster recording, for visualisation."""
        rng = np.random.default_rng(seed)
        W, wmax, ext = self._trial_setup(rng)
        theta = thetas_from_params(params, self.types)
        ph = phases if phases is not None else self.task.phases(rng)
        arr = np.array(ph, dtype=np.float64).T
        if plastic_override is not None:
            arr[1] = plastic_override
        c, t, cn = self.cfg, self.task, self.conn
        T = arr[0].sum()
        rec_every = int(round(rec_dt / c.dt))
        rec_rate = np.zeros(int(T / rec_dt) + 10)
        rec_t = np.zeros(4_000_000); rec_i = np.zeros(4_000_000, np.int32)
        W0 = W.copy()
        loss, tb, n_rec, n_spk = _run_one(
            int(rng.integers(1 << 30)), c.NE, c.NI, cn["pre"], cn["post"], cn["stype"],
            cn["out_ptr"], cn["in_ptr"], cn["in_syn"], W, wmax, theta, self.active, c.eta,
            ext, float(c.n_ext), c.w_ext, self.pat_k, rng.uniform(*getattr(t, "pat_rate", (0, 0))),
            arr[0], arr[1].astype(np.int64), arr[2].astype(np.int64), arr[3].astype(np.int64),
            arr[4].astype(np.int64), c.dt, c.tau_checker, c.rate_min, c.rate_max, t.task_id,
            getattr(t, "target", 10.0), getattr(t, "r_cut_fam", 20.0), getattr(t, "r_cut_nov", 10.0),
            rec_every, rec_rate, raster_n, rec_t, rec_i)
        return dict(loss=loss, t_blow=tb, rate=rec_rate[:n_rec], rec_dt=rec_dt,
                    spk_t=rec_t[:n_spk], spk_i=rec_i[:n_spk], W0=W0, W=W, stype=cn["stype"],
                    phases=ph, wmax=wmax, ext_rate=ext)
