"""
8_Performance_Metrics.py — Comprehensive Model Evaluation Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.ann_model import ATTACK_CLASSES
from utils.metrics import compute_classification_metrics, compute_roc_data
from utils.visualizations import confusion_matrix_heatmap, roc_curves_plot

from utils.theme import apply_custom_theme

st.set_page_config(page_title="📈 Performance Metrics", page_icon="📈", layout="wide")

apply_custom_theme()

st.markdown("# 📈 Performance Metrics")
st.markdown("Comprehensive evaluation of trained models with confusion matrix, ROC curves, and per-class metrics.")

# ──────────────────────────────────────────────
# Check for trained models
# ──────────────────────────────────────────────
has_ann = "ann_model" in st.session_state and "ann_results" in st.session_state
has_svm = "svm_model" in st.session_state and "svm_results" in st.session_state
has_data = "X_test" in st.session_state and "y_test" in st.session_state

# Fallback: try to populate X_test/y_test from preprocessing cache
if not has_data and "dataset_path" in st.session_state:
    from utils.data_loader import get_cached_preprocessing
    dataset_path = st.session_state["dataset_path"]
    dataset_type = st.session_state.get("dataset_type", "nsl-kdd")
    X_train, X_test, y_train, y_test, feats, _ = get_cached_preprocessing(dataset_path, dataset_type)
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["feature_names"] = feats
    has_data = True

if not has_data:
    st.warning("⚠️ No test data available. Preprocess data via **PCA Visualization** first.")
    st.stop()

if not has_ann and not has_svm:
    st.warning("⚠️ No trained model found. Train a model on the **ANN Training** page first.")
    st.stop()

y_test = st.session_state["y_test"]
X_test = st.session_state["X_test"]

# ──────────────────────────────────────────────
# Model selector
# ──────────────────────────────────────────────
available_models = []
if has_ann:
    available_models.append("ANN")
if has_svm:
    available_models.append("SVM")

selected_model = st.selectbox("Select Model to Evaluate", available_models)

if selected_model == "ANN":
    model = st.session_state["ann_model"]
    results = st.session_state["ann_results"]
elif selected_model == "SVM":
    model = st.session_state["svm_model"]
    results = st.session_state["svm_results"]

y_pred = results["y_pred"]

# ──────────────────────────────────────────────
# Overall Metrics
# ──────────────────────────────────────────────
st.markdown("## 🏆 Overall Metrics")
metrics = compute_classification_metrics(y_test, y_pred)

c1, c2, c3, c4 = st.columns(4)
metric_items = [
    ("Accuracy", metrics["accuracy"], "#00d4aa"),
    ("Precision", metrics["precision"], "#ffa502"),
    ("Recall", metrics["recall"], "#7c4dff"),
    ("F1 Score", metrics["f1"], "#ff4757"),
]
for col, (name, val, color) in zip([c1, c2, c3, c4], metric_items):
    with col:
        st.markdown(f"""<div class="metric-box">
            <div style="color:{color};font-size:2rem;font-weight:700">{val:.4f}</div>
            <div style="color:#8892b0">{name}</div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────
st.markdown("## 🔢 Confusion Matrix")
cm = metrics["confusion_matrix"]
fig_cm = confusion_matrix_heatmap(cm)
st.plotly_chart(fig_cm, use_container_width=True)

# ──────────────────────────────────────────────
# ROC Curves
# ──────────────────────────────────────────────
st.markdown("## 📉 ROC Curves (One-vs-Rest)")
try:
    y_proba = model.predict_proba(X_test)
    roc_data = compute_roc_data(y_test, y_proba)
    fig_roc = roc_curves_plot(roc_data)
    st.plotly_chart(fig_roc, use_container_width=True)
except Exception as e:
    st.warning(f"Could not compute ROC curves: {e}")

# ──────────────────────────────────────────────
# Per-Class Metrics
# ──────────────────────────────────────────────
st.markdown("## 📊 Per-Class Metrics")
report = metrics["report"]

rows = []
for cls in ATTACK_CLASSES:
    if cls in report:
        r = report[cls]
        rows.append({
            "Class": cls,
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-Score": f"{r['f1-score']:.4f}",
            "Support": int(r["support"]),
        })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# Clustering Metrics
# ──────────────────────────────────────────────
if "cluster_silhouette" in st.session_state:
    st.markdown("## 🎯 Clustering Metrics")
    sil = st.session_state["cluster_silhouette"]
    st.markdown(f"""<div class="metric-box" style="max-width:300px;">
        <div style="color:#ffa502;font-size:2rem;font-weight:700">{sil:.4f}</div>
        <div style="color:#8892b0">Silhouette Score</div>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# RL Metrics
# ──────────────────────────────────────────────
if "dqn_eval" in st.session_state:
    st.markdown("## 🤖 RL Agent Metrics")
    eval_results = st.session_state["dqn_eval"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Episode Reward", f"{eval_results['avg_reward']:.1f}")
    c2.metric("Reward Std Dev", f"{eval_results['std_reward']:.1f}")
    c3.metric("Correct Action Rate", f"{eval_results['avg_correct_rate']:.2%}")

# ──────────────────────────────────────────────
# Export Report
# ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📄 Export Full Classification Report"):
    report_text = pd.DataFrame(report).transpose()
    st.dataframe(report_text, use_container_width=True)

    csv = report_text.to_csv()
    st.download_button(
        label="📥 Download Report CSV",
        data=csv,
        file_name="classification_report.csv",
        mime="text/csv",
    )
