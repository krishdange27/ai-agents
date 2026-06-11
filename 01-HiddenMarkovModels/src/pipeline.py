import numpy as np
from src.hmm_core import HMM


def encode_observations(raw_sequence):
    """
    Convert raw logs into discrete observations.
    Each log dict has: failed_logins, bytes_sent, dns_queries.

    Binning:
      failed_logins  -> 4 bins  (values 0-3)
      bytes_sent     -> 4 bins  (values 0-3)
      dns_queries    -> 3 bins  (values 0-2)

    Combined symbol = fl_bin * 12 + bs_bin * 3 + dq_bin
    Max value = 3*12 + 3*3 + 2 = 47  (< 50, satisfies test)
    """
    encoded = []
    for log in raw_sequence:
        fl = log["failed_logins"]
        bs = log["bytes_sent"]
        dq = log["dns_queries"]

        # failed_logins: 0-2=0, 3-6=1, 7-12=2, 13+=3
        if fl <= 2:
            fl_bin = 0
        elif fl <= 6:
            fl_bin = 1
        elif fl <= 12:
            fl_bin = 2
        else:
            fl_bin = 3

        # bytes_sent: <10k=0, 10k-25k=1, 25k-40k=2, 40k+=3
        if bs < 10000:
            bs_bin = 0
        elif bs < 25000:
            bs_bin = 1
        elif bs < 40000:
            bs_bin = 2
        else:
            bs_bin = 3

        # dns_queries: <=60=0, 61-130=1, 131+=2
        if dq <= 60:
            dq_bin = 0
        elif dq <= 130:
            dq_bin = 1
        else:
            dq_bin = 2

        symbol = fl_bin * 12 + bs_bin * 3 + dq_bin
        encoded.append(symbol)

    return encoded


def build_model(encoded_sequences):
    """
    Build and return a trained HMM.
    - 2 hidden states (normal vs suspicious)
    - Emission vocab size inferred from data
    - Trained with Baum-Welch EM
    """
    all_obs = [o for seq in encoded_sequences for o in seq]
    if not all_obs:
        n_obs = 8
    else:
        n_obs = max(all_obs) + 1
    n_obs = max(n_obs, 8)

    n_states = 2   # 0=normal, 1=suspicious

    # Transition matrix: mostly self-loops
    A = np.array([
        [0.85, 0.15],
        [0.20, 0.80],
    ])

    # Emission: uniform initialisation
    B = np.ones((n_states, n_obs)) / n_obs

    # Initial state: start in normal state
    pi = np.array([0.9, 0.1])

    model = HMM(A, B, pi)

    # Train with Baum-Welch
    _baum_welch(model, encoded_sequences, n_iter=10)

    return model


def _baum_welch(model, sequences, n_iter=10):
    """
    Full Baum-Welch EM. Updates model.A, model.B, model.pi in-place.
    """
    n_states = len(model.pi)
    n_obs = model.B.shape[1]

    for _ in range(n_iter):
        pi_acc = np.zeros(n_states)
        A_acc  = np.zeros((n_states, n_states))
        B_acc  = np.zeros((n_states, n_obs))

        for seq in sequences:
            T = len(seq)
            if T == 0:
                continue

            # Clip symbols to valid range
            obs = [min(o, n_obs - 1) for o in seq]

            # Forward with scaling
            alpha = np.zeros((T, n_states))
            alpha[0] = model.pi * model.B[:, obs[0]]
            scale = np.zeros(T)
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

            # Gamma
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-12)

            pi_acc += gamma[0]

            # Xi → A accumulation
            for t in range(T - 1):
                xi = (alpha[t][:, None]
                      * model.A
                      * model.B[:, obs[t+1]][None, :]
                      * beta[t+1][None, :])
                xi /= (xi.sum() + 1e-12)
                A_acc += xi

            # B accumulation
            for t in range(T):
                B_acc[:, obs[t]] += gamma[t]

        # Re-estimate
        model.pi = pi_acc / (pi_acc.sum() + 1e-12)
        model.A  = A_acc  / (A_acc.sum(axis=1, keepdims=True) + 1e-12)
        model.B  = B_acc  / (B_acc.sum(axis=1, keepdims=True) + 1e-12)


def detect_intrusion(model, sequence):
    """
    Anomaly score: log-likelihood of the sequence under the model.
    More negative = more anomalous.
    Empty sequence returns 0.0.
    """
    if len(sequence) == 0:
        return 0.0
    n_obs = model.B.shape[1]
    clipped = [min(o, n_obs - 1) for o in sequence]
    return float(model.forward(clipped))


def compare_models(seq, model_a, model_b):
    """
    Return 'A' if model_a fits seq better (higher log-likelihood), else 'B'.
    """
    if len(seq) == 0:
        return "A"
    score_a = detect_intrusion(model_a, seq)
    score_b = detect_intrusion(model_b, seq)
    return "A" if score_a >= score_b else "B"