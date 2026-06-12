"""
demo.py — Intrusion Detection Agent
-------------------------------------
Trains the HMM on synthetic normal network logs (two regimes: baseline
and active sessions), then scores a mix of normal and anomalous sequences.

The agent flags sequences whose log-likelihood deviates significantly
from the distribution learned during training.

Run from the intrusion_detection/ folder:
    python3 demo.py
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from src.data import generate_raw_logs
from src.agent import encode_observations, build_model, detect_intrusion

# ── Anomalous sequences (realistic attack patterns) ───────────────────────────
# Each uses raw log values that encode to symbols rare/absent in normal traffic.

ANOMALOUS = {
    "Brute-force login attack": [
        # Rapid repeated login failures — failed_logins >> normal baseline
        {"failed_logins": 8,  "bytes_sent": 5200,  "dns_queries": 20},
        {"failed_logins": 11, "bytes_sent": 4800,  "dns_queries": 18},
        {"failed_logins": 9,  "bytes_sent": 5500,  "dns_queries": 22},
        {"failed_logins": 12, "bytes_sent": 4600,  "dns_queries": 19},
        {"failed_logins": 10, "bytes_sent": 5100,  "dns_queries": 21},
        {"failed_logins": 1,  "bytes_sent": 6000,  "dns_queries": 25},  # brief pause
        {"failed_logins": 9,  "bytes_sent": 5000,  "dns_queries": 20},
        {"failed_logins": 11, "bytes_sent": 4900,  "dns_queries": 18},
        {"failed_logins": 10, "bytes_sent": 5300,  "dns_queries": 22},
        {"failed_logins": 12, "bytes_sent": 4700,  "dns_queries": 19},
        {"failed_logins": 8,  "bytes_sent": 5400,  "dns_queries": 21},
        {"failed_logins": 11, "bytes_sent": 5000,  "dns_queries": 20},
    ],
    "Reconnaissance scan": [
        # Probing multiple services — elevated fails AND bytes together
        {"failed_logins": 5,  "bytes_sent": 18000, "dns_queries": 35},
        {"failed_logins": 6,  "bytes_sent": 20000, "dns_queries": 38},
        {"failed_logins": 4,  "bytes_sent": 17000, "dns_queries": 32},
        {"failed_logins": 6,  "bytes_sent": 19000, "dns_queries": 36},
        {"failed_logins": 5,  "bytes_sent": 21000, "dns_queries": 40},
        {"failed_logins": 1,  "bytes_sent": 6000,  "dns_queries": 22},  # brief pause
        {"failed_logins": 5,  "bytes_sent": 18500, "dns_queries": 34},
        {"failed_logins": 6,  "bytes_sent": 19500, "dns_queries": 37},
        {"failed_logins": 4,  "bytes_sent": 17500, "dns_queries": 33},
        {"failed_logins": 6,  "bytes_sent": 20500, "dns_queries": 39},
        {"failed_logins": 5,  "bytes_sent": 18000, "dns_queries": 35},
        {"failed_logins": 4,  "bytes_sent": 19000, "dns_queries": 36},
    ],
    "Data exfiltration": [
        # Sustained large outbound transfers — bytes_sent >> normal range
        {"failed_logins": 0,  "bytes_sent": 44000, "dns_queries": 22},
        {"failed_logins": 1,  "bytes_sent": 47000, "dns_queries": 25},
        {"failed_logins": 0,  "bytes_sent": 45500, "dns_queries": 20},
        {"failed_logins": 1,  "bytes_sent": 48000, "dns_queries": 24},
        {"failed_logins": 0,  "bytes_sent": 46000, "dns_queries": 21},
        {"failed_logins": 1,  "bytes_sent": 7000,  "dns_queries": 26},  # brief pause
        {"failed_logins": 0,  "bytes_sent": 45000, "dns_queries": 22},
        {"failed_logins": 1,  "bytes_sent": 47500, "dns_queries": 25},
        {"failed_logins": 0,  "bytes_sent": 44500, "dns_queries": 20},
        {"failed_logins": 1,  "bytes_sent": 48500, "dns_queries": 24},
        {"failed_logins": 0,  "bytes_sent": 46500, "dns_queries": 21},
        {"failed_logins": 1,  "bytes_sent": 45000, "dns_queries": 23},
    ],
}


def main():
    print("=" * 62)
    print("  Intrusion Detection Agent — HMM-based Anomaly Detector")
    print("=" * 62)

    # ── Train ─────────────────────────────────────────────────────
    print("\n[1] Generating training data (normal network traffic)...")
    raw_logs = generate_raw_logs("krishdange27", n_sequences=30)
    encoded  = [encode_observations(seq) for seq in raw_logs]
    avg_len  = sum(len(s) for s in encoded) // len(encoded)
    print(f"    {len(encoded)} sequences, avg length {avg_len} steps")

    print("\n[2] Training HMM (2 states, Baum-Welch EM)...")
    model = build_model(encoded)
    print("    Training complete.")

    # ── Compute threshold from training scores ─────────────────────
    normal_scores = [detect_intrusion(model, s) for s in encoded]
    mean_n = sum(normal_scores) / len(normal_scores)
    std_n  = math.sqrt(sum((s - mean_n)**2 for s in normal_scores) / len(normal_scores))
    threshold = mean_n - 1.5 * std_n

    # ── Score held-out normal sequences ───────────────────────────
    print(f"\n[3] Scoring normal sequences  (threshold = {threshold:.1f}):")
    print(f"    {'Sequence':<12}  {'Score':>8}  {'Deviation':>10}  Status")
    print("    " + "─" * 46)
    for i, score in enumerate(normal_scores[:5]):
        dev    = score - mean_n
        status = "ANOMALOUS ⚠" if score < threshold else "NORMAL ✓"
        print(f"    Normal {i+1:<5}  {score:>8.2f}  {dev:>+10.2f}  {status}")

    # ── Score anomalous sequences ──────────────────────────────────
    print(f"\n[4] Scoring anomalous sequences:")
    print(f"    {'Attack Type':<28}  {'Score':>8}  {'Deviation':>10}  Status")
    print("    " + "─" * 62)
    anomaly_scores = []
    for label, raw in ANOMALOUS.items():
        enc    = encode_observations(raw)
        score  = detect_intrusion(model, enc)
        dev    = score - mean_n
        status = "ANOMALOUS ⚠" if score < threshold else "NORMAL ✓"
        anomaly_scores.append(score)
        print(f"    {label:<28}  {score:>8.2f}  {dev:>+10.2f}  {status}")

    # ── Summary ───────────────────────────────────────────────────
    avg_anom = sum(anomaly_scores) / len(anomaly_scores)
    print(f"\n[5] Summary")
    print(f"    Normal traffic mean score  : {mean_n:.2f}")
    print(f"    Normal traffic std dev     : {std_n:.2f}")
    print(f"    Detection threshold        : {threshold:.2f}  (μ − 1.5σ)")
    print(f"    Anomalous sequences avg    : {avg_anom:.2f}")
    print(f"    Score gap                  : {mean_n - avg_anom:+.2f}")
    print()


if __name__ == "__main__":
    main()
