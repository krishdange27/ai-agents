# data.py
# Synthetic network log generator.
#
# Normal traffic has two natural regimes:
#   - Baseline activity  : minimal logins, low bytes, few DNS (most of the time)
#   - Active sessions    : slightly elevated across all metrics (periodic bursts)
#
# This gives the HMM two genuine states to learn, producing a richer
# emission distribution that makes anomalous sequences score distinctly.

import numpy as np


def get_seed(username):
    import hashlib
    return int(hashlib.md5(username.encode()).hexdigest(), 16) % (10**8)


def generate_raw_logs(username, n_sequences=30):
    """
    Generate synthetic normal network traffic logs with two regimes.

    Regime 0 (baseline, ~75% of steps):
      - failed_logins : 0-2   (Poisson λ=0.8)
      - bytes_sent    : 2-9k  (normal μ=5000)
      - dns_queries   : 5-40  (Poisson λ=18)

    Regime 1 (active session, ~25% of steps):
      - failed_logins : 0-3   (Poisson λ=1.5)
      - bytes_sent    : 8-22k (normal μ=14000)
      - dns_queries   : 20-70 (Poisson λ=40)

    The two regimes correspond naturally to the two HMM hidden states,
    giving Baum-Welch enough structure to learn distinct emissions.
    """
    seed = get_seed(username)
    rng  = np.random.RandomState(seed)

    logs = []
    for _ in range(n_sequences):
        seq_len = rng.randint(12, 25)
        seq     = []

        # Each sequence starts in a random regime and transitions between them
        state = 0 if rng.rand() < 0.75 else 1

        for _ in range(seq_len):
            # Transition between regimes
            if state == 0:
                state = 1 if rng.rand() < 0.15 else 0
            else:
                state = 0 if rng.rand() < 0.35 else 1

            if state == 0:
                # Baseline: low activity
                failed_logins = int(rng.poisson(0.8))
                bytes_sent    = int(np.clip(rng.normal(5000, 1500), 1000, 9500))
                dns_queries   = int(rng.poisson(18))
            else:
                # Active session: moderate elevated activity
                failed_logins = int(rng.poisson(1.5))
                bytes_sent    = int(np.clip(rng.normal(14000, 3000), 8000, 24000))
                dns_queries   = int(rng.poisson(40))

            seq.append({
                "failed_logins": failed_logins,
                "bytes_sent":    bytes_sent,
                "dns_queries":   dns_queries,
            })
        logs.append(seq)

    return logs
