"""
generate_synthetic.py — Generate a realistic synthetic NSL-KDD-style dataset.

This creates a CSV with the same column structure as NSL-KDD so the project
works out-of-the-box without needing the real dataset downloaded.

The synthetic data uses statistical distributions inspired by real NSL-KDD
to produce meaningful patterns that the ML models can learn from.
"""

import os
import numpy as np
import pandas as pd

# NSL-KDD column names (minus difficulty_level)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",
]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "smtp", "ftp", "ftp_data", "ssh", "telnet", "domain",
            "pop_3", "imap4", "private", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"]

# Attack labels for each class
ATTACK_LABELS = {
    "Normal": ["normal"],
    "DOS": ["back", "land", "neptune", "pod", "smurf", "teardrop"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf",
            "spy", "warezclient", "warezmaster"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit"],
}


def _generate_class_samples(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n samples for a given attack class."""
    # Class-specific parameter distributions
    profiles = {
        "Normal": {"duration_mu": 200, "src_mu": 2000, "dst_mu": 3000,
                   "serror": 0.01, "rerror": 0.01, "same_srv": 0.95},
        "DOS": {"duration_mu": 0, "src_mu": 500, "dst_mu": 0,
                "serror": 0.8, "rerror": 0.05, "same_srv": 0.1},
        "Probe": {"duration_mu": 0, "src_mu": 300, "dst_mu": 200,
                  "serror": 0.3, "rerror": 0.3, "same_srv": 0.3},
        "R2L": {"duration_mu": 500, "src_mu": 1500, "dst_mu": 5000,
                "serror": 0.05, "rerror": 0.1, "same_srv": 0.7},
        "U2R": {"duration_mu": 300, "src_mu": 1000, "dst_mu": 2000,
                "serror": 0.02, "rerror": 0.02, "same_srv": 0.8},
    }
    p = profiles[label]

    data = {
        "duration": np.abs(rng.normal(p["duration_mu"], 100, n)).astype(int),
        "protocol_type": rng.choice(PROTOCOLS, n),
        "service": rng.choice(SERVICES, n),
        "flag": rng.choice(FLAGS, n, p=[0.6, 0.1, 0.08, 0.05, 0.05,
                                         0.03, 0.03, 0.02, 0.02, 0.02]),
        "src_bytes": np.abs(rng.normal(p["src_mu"], 800, n)).astype(int),
        "dst_bytes": np.abs(rng.normal(p["dst_mu"], 1000, n)).astype(int),
        "land": rng.choice([0, 1], n, p=[0.99, 0.01]),
        "wrong_fragment": rng.choice([0, 1, 2, 3], n, p=[0.9, 0.05, 0.03, 0.02]),
        "urgent": rng.choice([0, 1], n, p=[0.98, 0.02]),
        "hot": rng.poisson(0.5, n),
        "num_failed_logins": rng.poisson(0.1, n),
        "logged_in": rng.choice([0, 1], n, p=[0.3, 0.7] if label == "Normal" else [0.6, 0.4]),
        "num_compromised": rng.poisson(0.2, n),
        "root_shell": rng.choice([0, 1], n, p=[0.95, 0.05] if label != "U2R" else [0.5, 0.5]),
        "su_attempted": rng.choice([0, 1, 2], n, p=[0.9, 0.05, 0.05]),
        "num_root": rng.poisson(0.1, n),
        "num_file_creations": rng.poisson(0.1, n),
        "num_shells": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "num_access_files": rng.poisson(0.1, n),
        "num_outbound_cmds": np.zeros(n, dtype=int),
        "is_host_login": rng.choice([0, 1], n, p=[0.99, 0.01]),
        "is_guest_login": rng.choice([0, 1], n, p=[0.98, 0.02]),
        "count": rng.poisson(50 if label == "DOS" else 10, n),
        "srv_count": rng.poisson(30 if label == "DOS" else 8, n),
        "serror_rate": np.clip(rng.normal(p["serror"], 0.1, n), 0, 1),
        "srv_serror_rate": np.clip(rng.normal(p["serror"], 0.1, n), 0, 1),
        "rerror_rate": np.clip(rng.normal(p["rerror"], 0.1, n), 0, 1),
        "srv_rerror_rate": np.clip(rng.normal(p["rerror"], 0.1, n), 0, 1),
        "same_srv_rate": np.clip(rng.normal(p["same_srv"], 0.15, n), 0, 1),
        "diff_srv_rate": np.clip(rng.normal(1 - p["same_srv"], 0.15, n), 0, 1),
        "srv_diff_host_rate": np.clip(rng.normal(0.1, 0.1, n), 0, 1),
        "dst_host_count": rng.integers(0, 256, n),
        "dst_host_srv_count": rng.integers(0, 256, n),
        "dst_host_same_srv_rate": np.clip(rng.normal(p["same_srv"], 0.2, n), 0, 1),
        "dst_host_diff_srv_rate": np.clip(rng.normal(1 - p["same_srv"], 0.2, n), 0, 1),
        "dst_host_same_src_port_rate": np.clip(rng.normal(0.3, 0.2, n), 0, 1),
        "dst_host_srv_diff_host_rate": np.clip(rng.normal(0.05, 0.05, n), 0, 1),
        "dst_host_serror_rate": np.clip(rng.normal(p["serror"], 0.1, n), 0, 1),
        "dst_host_srv_serror_rate": np.clip(rng.normal(p["serror"], 0.1, n), 0, 1),
        "dst_host_rerror_rate": np.clip(rng.normal(p["rerror"], 0.1, n), 0, 1),
        "dst_host_srv_rerror_rate": np.clip(rng.normal(p["rerror"], 0.1, n), 0, 1),
        "label": rng.choice(ATTACK_LABELS[label], n),
    }
    return pd.DataFrame(data)


def generate_synthetic_dataset(
    n_samples: int = 25000,
    output_path: str = "dataset/synthetic_nsl_kdd.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate and save a synthetic NSL-KDD-style dataset.

    Class distribution (approximate real-world imbalance):
      Normal: 40%, DOS: 35%, Probe: 15%, R2L: 7%, U2R: 3%
    """
    rng = np.random.default_rng(seed)
    class_ratios = {"Normal": 0.40, "DOS": 0.35, "Probe": 0.15, "R2L": 0.07, "U2R": 0.03}

    frames = []
    for cls, ratio in class_ratios.items():
        n = max(int(n_samples * ratio), 50)  # minimum 50 per class
        frames.append(_generate_class_samples(cls, n, rng))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    # Round float columns to 4 decimals for cleanliness
    float_cols = df.select_dtypes(include=[float]).columns
    df[float_cols] = df[float_cols].round(4)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved: {output_path}  ({len(df)} samples)")
    return df


if __name__ == "__main__":
    generate_synthetic_dataset()
