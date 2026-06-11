import numpy as np
import hashlib

def get_seed(username):
    return int(hashlib.md5(username.encode()).hexdigest(), 16) % (10**8)


def generate_raw_logs(username, n_sequences=30):
    seed = get_seed(username)
    np.random.seed(seed)

    logs = []
    for _ in range(n_sequences):
        seq = []
        for _ in range(np.random.randint(10, 25)):
            seq.append({
                "failed_logins": np.random.poisson(5),
                "bytes_sent": np.random.randint(1000, 50000),
                "dns_queries": np.random.randint(10, 200)
            })
        logs.append(seq)

    return logs