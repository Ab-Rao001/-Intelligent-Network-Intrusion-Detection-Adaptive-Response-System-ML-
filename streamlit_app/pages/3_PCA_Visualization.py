"""
3_PCA_Visualization.py — Principal Component Analysis
"""

import streamlit as st
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.visualizations import pca_scatter, pca_3d_scatter, explained_variance_plot
from utils.data_loader import get_cached_preprocessing, get_cached_pca

from utils.theme import apply_custom_theme

st.set_page_config(page_title="📐 PCA Visualization", page_icon="📐", layout="wide")

apply_custom_theme()

st.markdown("# 📐 PCA Visualization")
st.markdown("Reduce dimensionality and visualize high-dimensional network traffic data.")

if "dataset_path" not in st.session_state:
    st.warning("⚠️ No dataset loaded. Go to **Dataset Explorer** first.")
    st.stop()

# ──────────────────────────────────────────────
# PCA Configuration
# ──────────────────────────────────────────────
st.sidebar.markdown("### PCA Settings")
n_components = st.sidebar.slider("Number of components", 2, 20, 3)
view_mode = st.sidebar.radio("Visualization", ["2D", "3D"])

# ──────────────────────────────────────────────
# Run PCA
# ──────────────────────────────────────────────
if st.button("🚀 Run PCA", type="primary"):
    with st.spinner("Preprocessing and running PCA (using cached data)..."):
        dataset_path = st.session_state["dataset_path"]
        dataset_type = st.session_state.get("dataset_type", "nsl-kdd")

        # Preprocess
        X_train, X_test, y_train, y_test, feats, _ = get_cached_preprocessing(
            dataset_path, dataset_type
        )

        # PCA
        pca_data, variance, pipeline = get_cached_pca(X_train, n_components)

        # Store data in session so downstream pages (ANN, Prediction, Metrics) can access it
        st.session_state["pca_ran"] = True
        st.session_state["n_components"] = n_components
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["feature_names"] = feats

    st.success(f"✅ PCA complete! Reduced {X_train.shape[1]} features → {n_components} components")

# ──────────────────────────────────────────────
# Display Results
# ──────────────────────────────────────────────
if st.session_state.get("pca_ran", False):
    # Retrieve cached data quickly
    dataset_path = st.session_state["dataset_path"]
    dataset_type = st.session_state.get("dataset_type", "nsl-kdd")
    X_train, X_test, y_train, y_test, feats, _ = get_cached_preprocessing(dataset_path, dataset_type)
    pca_data, variance, pipeline = get_cached_pca(X_train, st.session_state["n_components"])
    y = y_train

    # Explained variance
    st.markdown("## 📊 Explained Variance")
    fig_var = explained_variance_plot(variance)
    st.plotly_chart(fig_var, use_container_width=True)

    total_var = np.sum(variance) * 100
    st.info(f"Total explained variance with {len(variance)} components: **{total_var:.1f}%**")

    # Scatter plot
    st.markdown("## 🗺️ PCA Scatter Plot")
    if view_mode == "3D" and pca_data.shape[1] >= 3:
        fig_scatter = pca_3d_scatter(pca_data, y)
    else:
        fig_scatter = pca_scatter(pca_data, y)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Component details
    with st.expander("📋 Component Details"):
        for i, v in enumerate(variance):
            st.write(f"**PC{i+1}**: {v*100:.2f}% variance explained")
else:
    st.info("👆 Click **Run PCA** to start dimensionality reduction.")
