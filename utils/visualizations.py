"""
visualizations.py — Shared Visualization Utilities

Plotly-based chart builders used across the Streamlit dashboard.
All functions return Plotly figure objects for easy integration.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ATTACK_CLASSES = ["Normal", "DOS", "Probe", "R2L", "U2R"]

# ──────────────────────────────────────────────
# Color palette — cybersecurity themed
# ──────────────────────────────────────────────
COLORS = {
    "Normal": "#00d4aa",
    "DOS": "#ff4757",
    "Probe": "#ffa502",
    "R2L": "#7c4dff",
    "U2R": "#ff6b81",
}
COLOR_LIST = ["#00d4aa", "#ff4757", "#ffa502", "#7c4dff", "#ff6b81",
              "#2ed573", "#1e90ff", "#ff6348", "#a29bfe", "#fd79a8"]

TEMPLATE = "plotly_dark"


def class_distribution_chart(labels: pd.Series | np.ndarray, title: str = "Class Distribution") -> go.Figure:
    """Bar chart of attack class distribution."""
    if isinstance(labels, np.ndarray):
        labels = pd.Series(labels)
    counts = labels.value_counts().reindex(ATTACK_CLASSES, fill_value=0)
    colors = [COLORS.get(c, "#888") for c in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors,
        text=counts.values, textposition="auto",
    ))
    fig.update_layout(
        title=title, template=TEMPLATE,
        xaxis_title="Attack Class", yaxis_title="Count",
        height=400,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation Heatmap") -> go.Figure:
    """Correlation matrix heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmid=0,
    ))
    fig.update_layout(title=title, template=TEMPLATE, height=600, width=700)
    return fig


def pca_scatter(pca_data: np.ndarray, labels: np.ndarray | None = None,
                title: str = "PCA Visualization") -> go.Figure:
    """2D scatter plot of PCA-reduced data."""
    df = pd.DataFrame(pca_data[:, :2], columns=["PC1", "PC2"])
    if labels is not None:
        if isinstance(labels[0], (int, np.integer)):
            df["Label"] = [ATTACK_CLASSES[int(l)] if int(l) < len(ATTACK_CLASSES) else str(l) for l in labels]
        else:
            df["Label"] = labels
        fig = px.scatter(df, x="PC1", y="PC2", color="Label",
                         color_discrete_map=COLORS, template=TEMPLATE, title=title)
    else:
        fig = px.scatter(df, x="PC1", y="PC2", template=TEMPLATE, title=title)
    fig.update_layout(height=500)
    return fig


def pca_3d_scatter(pca_data: np.ndarray, labels: np.ndarray | None = None,
                   title: str = "3D PCA Visualization") -> go.Figure:
    """3D scatter plot of PCA-reduced data (needs >= 3 components)."""
    if pca_data.shape[1] < 3:
        return pca_scatter(pca_data, labels, title)  # fallback to 2D

    df = pd.DataFrame(pca_data[:, :3], columns=["PC1", "PC2", "PC3"])
    if labels is not None:
        if isinstance(labels[0], (int, np.integer)):
            df["Label"] = [ATTACK_CLASSES[int(l)] if int(l) < len(ATTACK_CLASSES) else str(l) for l in labels]
        else:
            df["Label"] = labels
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="Label",
                            color_discrete_map=COLORS, template=TEMPLATE, title=title)
    else:
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                            template=TEMPLATE, title=title)
    fig.update_layout(height=600)
    return fig


def explained_variance_plot(variance_ratio: np.ndarray,
                            title: str = "PCA Explained Variance") -> go.Figure:
    """Bar + cumulative line for PCA explained variance."""
    cumulative = np.cumsum(variance_ratio)
    n = len(variance_ratio)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=list(range(1, n + 1)), y=variance_ratio,
        name="Individual", marker_color="#00d4aa",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)), y=cumulative,
        name="Cumulative", mode="lines+markers",
        marker_color="#ffa502", line=dict(width=2),
    ), secondary_y=True)
    fig.update_layout(
        title=title, template=TEMPLATE, height=400,
        xaxis_title="Principal Component",
    )
    fig.update_yaxes(title_text="Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative", secondary_y=True)
    return fig


def cluster_scatter(pca_data: np.ndarray, cluster_labels: np.ndarray,
                    title: str = "K-Means Clusters") -> go.Figure:
    """Scatter plot colored by cluster assignment."""
    df = pd.DataFrame(pca_data[:, :2], columns=["PC1", "PC2"])
    df["Cluster"] = cluster_labels.astype(str)
    fig = px.scatter(df, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=COLOR_LIST, template=TEMPLATE, title=title)
    fig.update_layout(height=500)
    return fig


def elbow_plot(elbow_data: list[tuple], title: str = "Elbow Method") -> go.Figure:
    """Line plot of K vs. inertia."""
    ks = [d[0] for d in elbow_data]
    inertias = [d[1] for d in elbow_data]
    fig = go.Figure(go.Scatter(
        x=ks, y=inertias, mode="lines+markers",
        marker=dict(size=10, color="#00d4aa"),
        line=dict(width=2, color="#00d4aa"),
    ))
    fig.update_layout(
        title=title, template=TEMPLATE, height=400,
        xaxis_title="Number of Clusters (K)", yaxis_title="Inertia",
    )
    return fig


def confusion_matrix_heatmap(cm: np.ndarray,
                             title: str = "Confusion Matrix") -> go.Figure:
    """Annotated heatmap for confusion matrix."""
    fig = go.Figure(go.Heatmap(
        z=cm, x=ATTACK_CLASSES, y=ATTACK_CLASSES,
        colorscale="Blues", text=cm, texttemplate="%{text}",
        textfont={"size": 14},
    ))
    fig.update_layout(
        title=title, template=TEMPLATE, height=500,
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    return fig


def roc_curves_plot(roc_data: dict, title: str = "ROC Curves (One-vs-Rest)") -> go.Figure:
    """Multi-class ROC curves."""
    fig = go.Figure()
    for i, cls in enumerate(ATTACK_CLASSES):
        if i in roc_data:
            fig.add_trace(go.Scatter(
                x=roc_data[i]["fpr"], y=roc_data[i]["tpr"],
                mode="lines", name=f"{cls} (AUC={roc_data[i]['auc']:.3f})",
                line=dict(color=COLOR_LIST[i], width=2),
            ))
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title=title, template=TEMPLATE, height=500,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    )
    return fig


def training_history_plot(history: dict, title: str = "Training History") -> go.Figure:
    """Plot loss and accuracy curves from Keras history."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    epochs = list(range(1, len(history["loss"]) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=history["loss"], name="Train Loss",
        mode="lines", line=dict(color="#ff4757"),
    ), row=1, col=1)
    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"], name="Val Loss",
            mode="lines", line=dict(color="#ffa502", dash="dash"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=history["accuracy"], name="Train Acc",
        mode="lines", line=dict(color="#00d4aa"),
    ), row=1, col=2)
    if "val_accuracy" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_accuracy"], name="Val Acc",
            mode="lines", line=dict(color="#7c4dff", dash="dash"),
        ), row=1, col=2)

    fig.update_layout(title=title, template=TEMPLATE, height=400)
    return fig


def reward_curve_plot(episode_rewards: list[float],
                      title: str = "RL Cumulative Reward") -> go.Figure:
    """Plot cumulative reward over episodes."""
    cumulative = np.cumsum(episode_rewards)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(episode_rewards) + 1)),
        y=episode_rewards, mode="lines+markers",
        name="Episode Reward", line=dict(color="#ffa502"),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative) + 1)),
        y=cumulative, mode="lines",
        name="Cumulative", line=dict(color="#00d4aa", width=2),
    ))
    fig.update_layout(
        title=title, template=TEMPLATE, height=400,
        xaxis_title="Episode", yaxis_title="Reward",
    )
    return fig


def feature_distribution_plot(df: pd.DataFrame, feature: str,
                              title: str | None = None) -> go.Figure:
    """Histogram of a single feature."""
    title = title or f"Distribution of {feature}"
    fig = px.histogram(df, x=feature, nbins=50,
                       color_discrete_sequence=["#00d4aa"],
                       template=TEMPLATE, title=title)
    fig.update_layout(height=350)
    return fig


def missing_values_chart(df: pd.DataFrame, title: str = "Missing Values") -> go.Figure:
    """Bar chart showing missing value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No missing values found! ✅",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=20, color="#00d4aa"))
        fig.update_layout(template=TEMPLATE, height=300)
        return fig

    fig = go.Figure(go.Bar(
        x=missing.index, y=missing.values,
        marker_color="#ff4757", text=missing.values, textposition="auto",
    ))
    fig.update_layout(
        title=title, template=TEMPLATE, height=400,
        xaxis_title="Feature", yaxis_title="Missing Count",
    )
    return fig
