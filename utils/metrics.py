"""
metrics.py — Evaluation Metrics Module

Provides unified metric computation for:
  - Classification (accuracy, precision, recall, F1, confusion matrix, ROC)
  - Clustering (silhouette score)
  - Reinforcement Learning (cumulative reward tracking)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize

ATTACK_CLASSES = ["Normal", "DOS", "Probe", "R2L", "U2R"]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard classification metrics.

    Returns:
        dict with accuracy, precision, recall, f1, report, confusion_matrix
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_true, y_pred,
            target_names=ATTACK_CLASSES,
            labels=range(len(ATTACK_CLASSES)),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def compute_roc_data(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 5) -> dict:
    """Compute per-class ROC curves (One-vs-Rest).

    Args:
        y_true: True labels (integer encoded)
        y_proba: Predicted probabilities, shape (n, n_classes)

    Returns:
        dict with keys per class: {class_idx: {"fpr", "tpr", "auc"}}
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    roc_data = {}

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            # No positive samples for this class
            roc_data[i] = {"fpr": [0, 1], "tpr": [0, 0], "auc": 0.0}
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[i] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    return roc_data


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for clustering results."""
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)


class RewardTracker:
    """Track cumulative and per-episode rewards for RL evaluation."""

    def __init__(self):
        self.episode_rewards: list[float] = []
        self.current_episode: list[float] = []

    def add_step_reward(self, reward: float):
        """Add a reward for the current step."""
        self.current_episode.append(reward)

    def end_episode(self):
        """End current episode and store cumulative reward."""
        if self.current_episode:
            self.episode_rewards.append(sum(self.current_episode))
            self.current_episode = []

    def get_cumulative_rewards(self) -> list[float]:
        """Return cumulative rewards over all episodes."""
        cumulative = []
        running = 0.0
        for r in self.episode_rewards:
            running += r
            cumulative.append(running)
        return cumulative

    def get_stats(self) -> dict:
        """Return reward statistics."""
        if not self.episode_rewards:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "total_episodes": 0}
        return {
            "mean": np.mean(self.episode_rewards),
            "std": np.std(self.episode_rewards),
            "min": np.min(self.episode_rewards),
            "max": np.max(self.episode_rewards),
            "total_episodes": len(self.episode_rewards),
        }
