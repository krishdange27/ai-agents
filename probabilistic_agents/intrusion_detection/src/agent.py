# agent.py
# Intrusion Detection Agent — HMM pipeline.
#
# Encodes raw network logs into discrete observation symbols,
# trains a 2-state HMM using Baum-Welch EM, and scores sequences
# by log-likelihood to detect anomalous behaviour.

import numpy as np
from src.hmm import HMM


def encode_observations(raw_sequence):
    """
    Convert raw network log entries into discrete observation symbols.

    Each log entry has three features:
      - failed_logins  : number of failed login attempts
      - bytes_sent     : bytes transferred in the interval
      - dns_queries    : number of DNS lookups made

    Each feature is binned independently and combined via mixed-radix encoding:
        symbol = fl_bin * 12 + bs_bin * 3 + dq_bin

    Bin boundaries are tuned so normal traffic clusters tightly around
    a small set of low-valued symbols, making anomalous patterns
    (elevated failures, high bytes, unusual DNS) clearly separable.
    """
    encoded = []
    for log in raw_sequence:
        fl = log["failed_logins"]
        bs = log["bytes_sent"]
        dq = log["dns_queries"]

        # failed_logins: 0-2=0, 3-6=1, 7-12=2, 13+=3
        if fl <= 2:   fl_bin = 0
        elif fl <= 6: fl_bin = 1
        elif fl <= 12:fl_bin = 2
        else:         fl_bin = 3

        # bytes_sent: <10k=0, 10-25k=1, 25-40k=2, 40k+=3
        if bs < 10000:   bs_bin = 0
        elif bs < 25000: bs_bin = 1
        elif bs < 40000: bs_bin = 2
        else:            bs_bin = 3

        # dns_queries: <=60=0, 61-130=1, 131+=2
        if dq <= 60:   dq_bin = 0
        elif dq <= 130:dq_bin = 1
        else:          dq_bin = 2

        encoded.append(fl_bin * 12 + bs_bin * 3 + dq_bin)

    return encoded


def build_model(encoded_sequences):
    """
    Build and train a 2-state HMM on normal network traffic sequences.

    State 0: low-activity normal behaviour (dominates most time steps)
    State 1: moderate-activity normal behaviour (occasional bursts)

    The two states are initialised with asymmetric emission distributions
    to prevent Baum-Welch from converging to a symmetric solution.
    """
    all_obs = [o for seq in encoded_sequences for o in seq]
    n_obs   = max(all_obs) + 1 if all_obs else 8
    n_obs   = max(n_obs, 8)

    n_states = 2

    # Asymmetric transition: state 0 is sticky (normal baseline),
    # state 1 transitions more freely (captures activity bursts)
    A = np.array([
        [0.92, 0.08],
        [0.30, 0.70],
    ])

    # Asymmetric emission initialisation — critical for breaking symmetry.
    # State 0 strongly favours symbol 0 (lowest-activity normal).
    # State 1 spreads probability more evenly across higher symbols.
    B = np.ones((n_states, n_obs)) * 0.01
    # State 0: concentrated on symbol 0
    B[0, 0] = 0.80
    if n_obs > 3:  B[0, 3]  = 0.10
    if n_obs > 12: B[0, 12] = 0.05
    if n_obs > 15: B[0, 15] = 0.02
    # State 1: more spread, higher symbols weighted
    if n_obs > 3:  B[1, 3]  = 0.30
    if n_obs > 12: B[1, 12] = 0.35
    if n_obs > 15: B[1, 15] = 0.25
    B[1, 0] = 0.05
    # Normalise rows
    B = B / B.sum(axis=1, keepdims=True)

    pi = np.array([0.90, 0.10])

    model = HMM(A, B, pi)
    _baum_welch(model, encoded_sequences, n_iter=20)
    return model


def _baum_welch(model, sequences, n_iter=20):
    """Baum-Welch EM with scaled forward-backward to prevent underflow."""
    n_states = len(model.pi)
    n_obs    = model.B.shape[1]

    for _ in range(n_iter):
        pi_acc = np.zeros(n_states)
        A_acc  = np.zeros((n_states, n_states))
        B_acc  = np.zeros((n_states, n_obs))

        for seq in sequences:
            T = len(seq)
            if T == 0:
                continue
            obs = [min(o, n_obs - 1) for o in seq]

            # Forward with scaling
            alpha = np.zeros((T, n_states))
            scale = np.zeros(T)
            alpha[0] = model.pi * model.B[:, obs[0]]
            scale[0] = alpha[0].sum() + 1e-12
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t-1] @ model.A) * model.B[:, obs[t]]
                scale[t] = alpha[t].sum() + 1e-12
                alpha[t] /= scale[t]

            # Backward with scaling
            beta = np.ones((T, n_states))
            for t in reversed(range(T - 1)):
                beta[t] = (model.A * model.B[:, obs[t+1]]) @ beta[t+1]
                beta[t] /= scale[t+1]

            # Gamma and xi
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-12)

            pi_acc += gamma[0]
            for t in range(T - 1):
                xi = (alpha[t][:, None] * model.A
                      * model.B[:, obs[t+1]][None, :] * beta[t+1][None, :])
                xi /= (xi.sum() + 1e-12)
                A_acc += xi
            for t in range(T):
                B_acc[:, obs[t]] += gamma[t]

        model.pi = pi_acc / (pi_acc.sum() + 1e-12)
        model.A  = A_acc  / (A_acc.sum(axis=1, keepdims=True) + 1e-12)
        model.B  = B_acc  / (B_acc.sum(axis=1, keepdims=True) + 1e-12)


def detect_intrusion(model, sequence):
    """
    Compute the anomaly score for a sequence.
    Returns log P(O | model) — lower means more anomalous.
    Empty sequences return 0.0.
    """
    if len(sequence) == 0:
        return 0.0
    n_obs   = model.B.shape[1]
    clipped = [min(o, n_obs - 1) for o in sequence]
    return float(model.forward(clipped))


def compare_models(seq, model_a, model_b):
    """Return 'A' if model_a fits seq better, else 'B'."""
    if len(seq) == 0:
        return "A"
    return "A" if detect_intrusion(model_a, seq) >= detect_intrusion(model_b, seq) else "B"
