"""
utils/model_loader.py — Cached model loading functions
"""

import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.ann_model import ANNClassifier
from reinforcement_learning.dqn_agent import DQNAgent, NetworkDefenseEnv

@st.cache_resource(show_spinner=False)
def load_cached_ann(model_path: str):
    """Loads ANN model, cached globally."""
    if not os.path.exists(model_path):
        return None
    ann = ANNClassifier(input_dim=1) # Dummy dim, overwritten on load
    try:
        ann.load(model_path)
        return ann
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_cached_dqn(model_path: str):
    """Loads DQN model, cached globally."""
    if not os.path.exists(model_path + ".zip"):
        return None
    env = NetworkDefenseEnv(max_steps=100)
    agent = DQNAgent(env, verbose=0)
    try:
        agent.load(model_path)
        return agent
    except Exception:
        return None
