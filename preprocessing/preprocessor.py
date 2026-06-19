"""
preprocessor.py — Data Preprocessing Pipeline for Network Intrusion Detection

Handles:
  - Loading NSL-KDD / CICIDS2017 datasets (CSV)
  - Missing value imputation
  - Label encoding of categorical features
  - MinMax scaling of numerical features
  - Stratified train/test splitting
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# ──────────────────────────────────────────────
# NSL-KDD column names (the raw files have no header)
# ──────────────────────────────────────────────
NSL_KDD_COLUMNS = [
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
    "label", "difficulty_level",
]

# ──────────────────────────────────────────────
# Attack → 5-class mapping for NSL-KDD
# ──────────────────────────────────────────────
ATTACK_MAP = {
    "normal": "Normal",
    # DOS attacks
    "back": "DOS", "land": "DOS", "neptune": "DOS", "pod": "DOS",
    "smurf": "DOS", "teardrop": "DOS", "mailbomb": "DOS", "apache2": "DOS",
    "processtable": "DOS", "udpstorm": "DOS",
    # Probe attacks
    "ipsweep": "DOS", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "worm": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}

# Fix: ipsweep should be Probe
ATTACK_MAP["ipsweep"] = "Probe"

CICIDS_ATTACK_MAP = {
    "Normal Traffic": "Normal",
    "BENIGN": "Normal",
    "DoS Hulk": "DOS",
    "DoS GoldenEye": "DOS",
    "DoS slowloris": "DOS",
    "DoS Slowhttptest": "DOS",
    "Heartbleed": "DOS",
    "DDoS": "DOS",
    "PortScan": "Probe",
    "FTP-Patator": "R2L",
    "SSH-Patator": "R2L",
    "Web Attack - Brute Force": "U2R",
    "Web Attack - XSS": "U2R",
    "Web Attack - Sql Injection": "U2R",
    "Infiltration": "U2R",
    "Bot": "U2R"
}

# 5-class label order
ATTACK_CLASSES = ["Normal", "DOS", "Probe", "R2L", "U2R"]


class DataPreprocessor:
    """End-to-end preprocessing pipeline for network intrusion datasets."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = MinMaxScaler()
        self.feature_names: list[str] = []
        self.target_encoder = LabelEncoder()

    # ──────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────
    def load_dataset(self, filepath: str, dataset_type: str = "nsl-kdd", n_samples: int | None = None) -> pd.DataFrame:
        """Load a CSV dataset and return a DataFrame.

        Args:
            filepath: Path to the CSV file.
            dataset_type: 'nsl-kdd' or 'cicids' — determines parsing logic.
            n_samples: Optional number of samples to read (useful for huge datasets like CICIDS).
        """
        if dataset_type == "nsl-kdd":
            df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS, nrows=n_samples)
            # Drop difficulty column (not needed for ML)
            if "difficulty_level" in df.columns:
                df.drop(columns=["difficulty_level"], inplace=True)
            # Map granular labels → 5 classes
            df["label"] = df["label"].str.strip().str.lower().map(ATTACK_MAP).fillna("Normal")
        else:
            # CICIDS — already has headers
            df = pd.read_csv(filepath, nrows=n_samples)
            # Standardize label column name
            label_col = [c for c in df.columns if "label" in c.lower() or "attack type" in c.lower()]
            if label_col:
                df.rename(columns={label_col[-1]: "label"}, inplace=True)
            # Map granular labels → 5 classes
            if "label" in df.columns:
                df["label"] = df["label"].str.strip().map(CICIDS_ATTACK_MAP).fillna("Normal")
        return df

    # ──────────────────────────────────────────
    # Cleaning
    # ──────────────────────────────────────────
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Rigorously clean missing numerics, invalid strings, infs, and outliers."""
        # 1. Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Try to coerce known numeric columns (everything except label)
        cols_to_coerce = [c for c in df.columns if c != "label"]
        for col in cols_to_coerce:
            if df[col].dtype == object:
                # Attempt numeric coercion; if it fails entirely, it remains an object,
                # but invalid numeric strings become NaN.
                df[col] = pd.to_numeric(df[col], errors='ignore')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.difference(["label"])

        # 3. Impute numeric columns with median, and cap outliers
        for col in numeric_cols:
            # Force coercion to catch lingering weird strings like "infinity"
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
            
            # Cap extreme outliers (1st and 99th percentile) to prevent scaler squashing
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower_bound, upper_bound)

        # 4. Impute categorical columns with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)

        # Drop rows where label is still missing
        df.dropna(subset=["label"], inplace=True)
        return df

    # ──────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode all categorical feature columns (not the target)."""
        categorical_cols = df.select_dtypes(include=["object"]).columns.difference(["label"])
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels to integers."""
        self.target_encoder.fit(ATTACK_CLASSES)
        return self.target_encoder.transform(y)

    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer labels back to class names."""
        return self.target_encoder.inverse_transform(y_encoded)

    # ──────────────────────────────────────────
    # Scaling
    # ──────────────────────────────────────────
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray):
        """MinMax-scale features (fit on train, transform both)."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    # ──────────────────────────────────────────
    # Full Pipeline
    # ──────────────────────────────────────────
    def run_pipeline(
        self,
        filepath: str,
        dataset_type: str = "nsl-kdd",
        test_size: float = 0.2,
        random_state: int = 42,
        n_samples: int | None = None,
    ):
        """Execute the full preprocessing pipeline.

        Returns:
            X_train, X_test, y_train, y_test, feature_names, df_clean
        """
        # 1. Load
        df = self.load_dataset(filepath, dataset_type, n_samples)

        # 2. Clean
        df = self.handle_missing_values(df)

        # 3. Encode features
        df = self.encode_features(df)

        # 4. Split features / target
        X = df.drop(columns=["label"])
        y = df["label"]
        self.feature_names = list(X.columns)

        # 5. Encode target
        y_encoded = self.encode_target(y)

        # 6. Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=test_size,
            random_state=random_state, stratify=y_encoded,
        )

        # 7. Scale
        X_train, X_test = self.scale_features(X_train, X_test)

        return X_train, X_test, y_train, y_test, self.feature_names, df

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────
    def save(self, directory: str = "saved_models"):
        """Save scaler and encoders."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(directory, "label_encoders.pkl"))
        joblib.dump(self.target_encoder, os.path.join(directory, "target_encoder.pkl"))

    def load(self, directory: str = "saved_models"):
        """Load scaler and encoders."""
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(directory, "label_encoders.pkl"))
        self.target_encoder = joblib.load(os.path.join(directory, "target_encoder.pkl"))


# ──────────────────────────────────────────────
# CLI quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/synthetic_nsl_kdd.csv"
    pp = DataPreprocessor()
    X_tr, X_te, y_tr, y_te, feats, df = pp.run_pipeline(path)
    print(f"Preprocessing complete")
    print(f"   Train: {X_tr.shape}  |  Test: {X_te.shape}")
    print(f"   Features: {len(feats)}  |  Classes: {np.unique(y_tr)}")
    pp.save()
    print("   Models saved to saved_models/")
