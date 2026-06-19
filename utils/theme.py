import streamlit as st

def apply_custom_theme():
    """
    Applies a highly cinematic, dark, and moody cybersecurity aesthetic
    to the Streamlit app globally using custom CSS injections.
    """
    custom_css = """
    <style>
        /* 
         * GLOBAL APP BACKGROUND
         * Deep, moody radial gradient to feel like a high-end command center.
         */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #111a22 0%, #080c10 60%, #030507 100%) !important;
            color: #d1d9e0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* 
         * SIDEBAR
         * Solid dark with a subtle right-border glow.
         */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0f14 0%, #05080a 100%) !important;
            border-right: 1px solid rgba(0, 212, 170, 0.1) !important;
        }

        /* 
         * TYPOGRAPHY
         * Crisp, muted, and serious.
         */
        h1, h2, h3, h4, h5, h6 {
            color: #f0f4f8 !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 2.2rem !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem !important;
        }

        p, span, div {
            color: #a0abb6;
        }

        /* 
         * GLASSMORPHISM METRICS / CARDS
         * Used for metrics, stats, and logical groupings.
         */
        [data-testid="metric-container"], .cluster-stat, .train-card, .stat-box, .metric-card, .feature-card {
            background: rgba(15, 22, 30, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
            padding: 20px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        [data-testid="metric-container"]:hover, .cluster-stat:hover, .train-card:hover, .stat-box:hover, .metric-card:hover, .feature-card:hover {
            border-color: rgba(0, 212, 170, 0.3) !important;
            transform: translateY(-2px);
        }

        /* Metric text overrides */
        [data-testid="metric-container"] label {
            color: #7b8895 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #00d4aa !important;
            font-weight: 700 !important;
        }

        /* 
         * BUTTONS
         * Minimalist outline, glowing hover effect.
         */
        .stButton>button {
            background: rgba(0, 212, 170, 0.05) !important;
            border: 1px solid rgba(0, 212, 170, 0.3) !important;
            color: #00d4aa !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 10px rgba(0, 212, 170, 0.0) !important;
        }

        .stButton>button:hover {
            background: rgba(0, 212, 170, 0.15) !important;
            border: 1px solid rgba(0, 212, 170, 0.6) !important;
            box-shadow: 0 0 15px rgba(0, 212, 170, 0.2) !important;
            transform: translateY(-1px) !important;
            color: #ffffff !important;
        }

        /* 
         * PRIMARY BUTTON OVERRIDE
         */
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #00d4aa 0%, #00a383 100%) !important;
            color: #000000 !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3) !important;
        }
        .stButton>button[kind="primary"]:hover {
            background: linear-gradient(135deg, #00f5c4 0%, #00c49e 100%) !important;
            box-shadow: 0 6px 20px rgba(0, 212, 170, 0.5) !important;
            color: #000000 !important;
        }

        /* 
         * TABS
         * Clean underline style.
         */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: transparent !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: #7b8895 !important;
            font-weight: 500 !important;
            border-bottom: 2px solid transparent !important;
            padding-bottom: 0.5rem !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #f0f4f8 !important;
            border-bottom: 2px solid #00d4aa !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: rgba(255,255,255,0.02) !important;
            border-radius: 6px;
            color: #a0abb6 !important;
        }
        .streamlit-expanderContent {
            border: 1px solid rgba(255,255,255,0.05);
            border-top: none;
            background-color: rgba(0,0,0,0.2) !important;
        }

        /* Progress Bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #00d4aa 0%, #0077ff 100%) !important;
        }

        /* Dataframes & Tables */
        .stDataFrame {
            border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
