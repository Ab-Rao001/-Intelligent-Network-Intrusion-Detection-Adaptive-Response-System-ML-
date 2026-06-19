"""
app.py — Main Streamlit Entry Point (Home Page)

Run with:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import os
import sys

# ──────────────────────────────────────────────
# Add project root to path so imports work
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.theme import apply_custom_theme

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ AI Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
apply_custom_theme()

st.markdown("""
<style>
    /* Specific overrides for app.py */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.15);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #8892b0;
        text-align: center;
        margin-top: 8px;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(124, 77, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        min-height: 180px;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }
    .feature-title {
        color: #00d4aa;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .feature-desc {
        color: #8892b0;
        font-size: 0.9rem;
    }
    
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4aa, #7c4dff, transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Hero Section
# ──────────────────────────────────────────────
st.markdown('<p class="hero-title">🛡️ AI Network Intrusion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Intelligent Detection • Adaptive Response • Real-Time Analysis</p>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Quick Stats
# ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #00d4aa; font-size: 2rem; font-weight: 700;">5</div>
        <div style="color: #8892b0;">Attack Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #ffa502; font-size: 2rem; font-weight: 700;">3</div>
        <div style="color: #8892b0;">ML Paradigms</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #7c4dff; font-size: 2rem; font-weight: 700;">5</div>
        <div style="color: #8892b0;">Defense Actions</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #ff4757; font-size: 2rem; font-weight: 700;">&lt;1s</div>
        <div style="color: #8892b0;">Prediction Latency</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Features Grid
# ──────────────────────────────────────────────
st.markdown("## 🧩 System Modules")
st.write("")

features = [
    ("📊", "Dataset Explorer", "Upload & explore NSL-KDD / CICIDS2017 datasets"),
    ("🔍", "EDA Dashboard", "Feature distributions, correlations, missing values"),
    ("📐", "PCA Visualization", "Dimensionality reduction with explained variance"),
    ("🎯", "K-Means Clustering", "Discover hidden attack patterns in traffic"),
    ("🧠", "ANN Training", "Train deep neural network for attack classification"),
    ("⚡", "Prediction Interface", "Real-time attack type prediction"),
    ("🤖", "RL Response Module", "DQN-based adaptive defense recommendations"),
    ("📈", "Performance Metrics", "Accuracy, ROC, confusion matrix analysis"),
]

cols = st.columns(4)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 4]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    if i == 3:
        cols = st.columns(4)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────
st.markdown("## 🏗️ System Architecture")
st.write("")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #00d4aa;">ML Pipeline</h4>
        <p style="color: #8892b0;">
        1. <b>Preprocessing</b> — Clean, encode, scale<br>
        2. <b>PCA</b> — Dimensionality reduction<br>
        3. <b>K-Means</b> — Unsupervised clustering<br>
        4. <b>ANN/SVM</b> — Supervised classification<br>
        5. <b>DQN</b> — Reinforcement learning response
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #7c4dff;">Attack Classes</h4>
        <p style="color: #8892b0;">
        🟢 <b>Normal</b> — Legitimate traffic<br>
        🔴 <b>DOS</b> — Denial of Service<br>
        🟡 <b>Probe</b> — Surveillance/scanning<br>
        🟣 <b>R2L</b> — Remote to Local<br>
        🩷 <b>U2R</b> — User to Root
        </p>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Getting started
# ──────────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🚀 Getting Started")
st.info(
    "👈 Use the **sidebar** to navigate between modules. "
    "Start with **Dataset Explorer** to upload or generate your dataset, "
    "then proceed through the ML pipeline."
)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #4a5568;'>"
    "Built with ❤️ using Streamlit • TensorFlow • Stable-Baselines3"
    "</p>",
    unsafe_allow_html=True,
)
