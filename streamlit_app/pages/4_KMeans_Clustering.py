"""
4_KMeans_Clustering.py — K-Means Clustering Module
"""

import streamlit as st
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustering.cluster_model import ClusteringPipeline
from utils.visualizations import cluster_scatter, elbow_plot
from utils.metrics import compute_silhouette
from utils.data_loader import get_cached_preprocessing, get_cached_pca

from utils.theme import apply_custom_theme

st.set_page_config(page_title="🎯 K-Means Clustering", page_icon="🎯", layout="wide")

apply_custom_theme()

st.markdown("# 🎯 K-Means Clustering")
st.markdown("Discover hidden attack groups in network traffic using unsupervised learning.")

if not st.session_state.get("pca_ran", False):
    st.warning("⚠️ Run PCA first on the **PCA Visualization** page.")
    st.stop()

dataset_path = st.session_state["dataset_path"]
dataset_type = st.session_state.get("dataset_type", "nsl-kdd")
X_train, _, _, _, _, _ = get_cached_preprocessing(dataset_path, dataset_type)
pca_data, _, _ = get_cached_pca(X_train, st.session_state["n_components"])

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
st.sidebar.markdown("### Clustering Settings")
n_clusters = st.sidebar.slider("Number of clusters (K)", 2, 10, 5)
show_elbow = st.sidebar.checkbox("Show Elbow Plot", value=True)

# ──────────────────────────────────────────────
# Elbow Method
# ──────────────────────────────────────────────
if show_elbow:
    st.markdown("## 📈 Elbow Method")
    with st.spinner("Computing elbow curve..."):
        try:
            elbow_data = ClusteringPipeline.elbow_method(pca_data, k_range=range(2, 11))
            fig_elbow = elbow_plot(elbow_data)
            st.plotly_chart(fig_elbow, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed to compute Elbow Curve: {str(e)}")

# ──────────────────────────────────────────────
# Run K-Means
# ──────────────────────────────────────────────
if st.button("🎯 Run K-Means Clustering", type="primary"):
    with st.spinner(f"Clustering with K={n_clusters}..."):
        try:
            pipeline = ClusteringPipeline(n_pca_components=pca_data.shape[1], n_clusters=n_clusters)
            pipeline.pca_data = pca_data
            labels = pipeline.fit_kmeans(pca_data)
            sil = pipeline.silhouette

            st.session_state["cluster_labels"] = labels
            st.session_state["cluster_silhouette"] = sil
            st.session_state["cluster_pipeline"] = pipeline

            st.success(f"✅ Clustering complete! Silhouette Score: **{sil:.4f}**")
        except Exception as e:
            st.error(f"❌ K-Means Clustering Failed: {str(e)}")

# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────
if "cluster_labels" in st.session_state:
    labels = st.session_state["cluster_labels"]
    sil = st.session_state["cluster_silhouette"]

    # Stats
    st.markdown("## 📊 Clustering Results")
    unique, counts = np.unique(labels, return_counts=True)

    cols = st.columns(len(unique) + 1)
    with cols[0]:
        st.markdown(f"""<div class="cluster-stat">
            <div style="color:#00d4aa;font-size:1.5rem;font-weight:700">{sil:.4f}</div>
            <div style="color:#8892b0">Silhouette Score</div>
        </div>""", unsafe_allow_html=True)

    for i, (cluster_id, count) in enumerate(zip(unique, counts)):
        with cols[i + 1]:
            st.markdown(f"""<div class="cluster-stat">
                <div style="color:#ffa502;font-size:1.5rem;font-weight:700">{count:,}</div>
                <div style="color:#8892b0">Cluster {cluster_id}</div>
            </div>""", unsafe_allow_html=True)

    # Scatter
    st.markdown("## 🗺️ Cluster Visualization")
    fig_cluster = cluster_scatter(pca_data, labels)
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Save
    if st.button("💾 Save Clustering Models"):
        pipeline = st.session_state["cluster_pipeline"]
        save_dir = os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"))
        pipeline.save(save_dir)
        st.success(f"✅ Models saved to `saved_models/{st.session_state.get("dataset_type", "nsl-kdd")}/`")

else:
    st.info("👆 Click **Run K-Means Clustering** to start.")
