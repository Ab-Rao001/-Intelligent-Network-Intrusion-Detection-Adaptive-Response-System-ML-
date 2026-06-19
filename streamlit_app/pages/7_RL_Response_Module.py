"""
7_RL_Response_Module.py — Reinforcement Learning Defense Recommendations
"""

import streamlit as st
import numpy as np
import os
import sys
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reinforcement_learning.dqn_agent import (
    NetworkDefenseEnv, DQNAgent, ATTACK_CLASSES,
    ACTION_NAMES, OPTIMAL_ACTIONS, SB3_AVAILABLE,
)
from utils.visualizations import reward_curve_plot

from utils.theme import apply_custom_theme

st.set_page_config(page_title="🤖 RL Response", page_icon="🤖", layout="wide")

apply_custom_theme()

st.markdown("# 🤖 RL Response Module")
st.markdown("Train a DQN agent to learn optimal defense actions for detected attacks.")

if not SB3_AVAILABLE:
    st.error("❌ `stable-baselines3` is not installed. Run: `pip install stable-baselines3[extra]`")
    st.stop()

# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────
st.sidebar.markdown("### RL Settings")
total_timesteps = st.sidebar.slider("Training Timesteps", 5000, 50000, 20000, step=5000)
max_steps = st.sidebar.slider("Steps per Episode", 50, 500, 200, step=50)
n_eval_episodes = st.sidebar.slider("Evaluation Episodes", 5, 50, 10)

# ──────────────────────────────────────────────
# Action Reference
# ──────────────────────────────────────────────
with st.expander("📖 Action Reference & Reward Structure", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Available Actions")
        for idx, name in ACTION_NAMES.items():
            st.write(f"**{idx}**: {name}")
    with col2:
        st.markdown("#### Reward Structure")
        st.write("✅ **+20** — Correct response")
        st.write("⚠️ **-5** — False alarm (acting on Normal)")
        st.write("❌ **-10** — Wrong/insufficient response")

    st.markdown("#### Optimal Action Mapping")
    for cls_idx, action_idx in OPTIMAL_ACTIONS.items():
        st.write(f"**{ATTACK_CLASSES[cls_idx]}** → {ACTION_NAMES[action_idx]}")

# ──────────────────────────────────────────────
# Train DQN
# ──────────────────────────────────────────────
st.markdown("## 🏋️ Train DQN Agent")

if st.button("🚀 Train DQN Agent", type="primary"):
    st.session_state["dqn_training_active"] = True
    st.session_state["dqn_training_done"] = False
    st.session_state["dqn_progress"] = {"step": 0, "total": total_timesteps}
    st.rerun()

if st.session_state.get("dqn_training_active", False):
    st.info("🔄 DQN Training running in background...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def dqn_train_task():
        env = NetworkDefenseEnv(max_steps=max_steps)
        agent = DQNAgent(env, verbose=0)

        def dqn_progress_cb(step, total):
            st.session_state["dqn_progress"] = {"step": step, "total": total}

        agent.train(total_timesteps=total_timesteps, progress_callback=dqn_progress_cb)

        st.session_state["dqn_agent"] = agent
        st.session_state["dqn_env"] = env
        st.session_state["dqn_training_active"] = False
        st.session_state["dqn_training_done"] = True

    # Start thread
    if "dqn_thread" not in st.session_state or not st.session_state["dqn_thread"].is_alive():
        thread = threading.Thread(target=dqn_train_task)
        add_script_run_ctx(thread)
        thread.start()
        st.session_state["dqn_thread"] = thread

    # Polling loop
    while st.session_state.get("dqn_training_active", True):
        prog = st.session_state.get("dqn_progress", {})
        step = prog.get("step", 0)
        total = prog.get("total", total_timesteps)
        
        pct = min(step / max(total, 1), 1.0)
        progress_bar.progress(pct)
        status_text.text(f"Timestep {step} / {total}")
        time.sleep(1)
        st.rerun()

if st.session_state.get("dqn_training_done", False):
    st.success("✅ DQN training complete!")

# ──────────────────────────────────────────────
# Evaluate & Display Results
# ──────────────────────────────────────────────
if "dqn_agent" in st.session_state:
    agent = st.session_state["dqn_agent"]

    st.markdown("## 📊 Evaluation")

    if st.button("🔄 Run Evaluation"):
        with st.spinner("Evaluating agent..."):
            eval_results = agent.evaluate(n_episodes=n_eval_episodes)
            st.session_state["dqn_eval"] = eval_results

    if "dqn_eval" in st.session_state:
        eval_results = st.session_state["dqn_eval"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Reward", f"{eval_results['avg_reward']:.1f}")
        c2.metric("Reward Std", f"{eval_results['std_reward']:.1f}")
        c3.metric("Correct Rate", f"{eval_results['avg_correct_rate']:.2%}")

        # Reward curve
        fig_reward = reward_curve_plot(eval_results["episode_rewards"])
        st.plotly_chart(fig_reward, use_container_width=True)

    # ──────────────────────────────────────────
    # Recommended Actions
    # ──────────────────────────────────────────
    st.markdown("## 🎯 Recommended Actions per Attack Class")
    predictions = agent.predict_all()

    cols = st.columns(5)
    for i, (cls, info) in enumerate(predictions.items()):
        with cols[i]:
            is_correct = info["is_correct"]
            css = "action-correct" if is_correct else "action-wrong"
            icon = "✅" if is_correct else "❌"
            color = "#00d4aa" if is_correct else "#ff4757"

            st.markdown(f"""
            <div class="action-card {css}">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="color:{color};font-size:1.1rem;font-weight:700;margin:8px 0;">
                    {cls}
                </div>
                <div style="color:#e6f1ff;font-size:0.95rem;">
                    → {info['action_name']}
                </div>
                <div style="color:#8892b0;font-size:0.8rem;margin-top:4px;">
                    Optimal: {info['optimal_action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # Interactive Prediction
    # ──────────────────────────────────────────
    st.markdown("## 🔮 Interactive Response")
    selected_class = st.selectbox("Select detected attack class:", ATTACK_CLASSES)
    class_idx = ATTACK_CLASSES.index(selected_class)

    if st.button("🛡️ Get Defense Recommendation", type="primary"):
        action_idx, action_name = agent.predict(class_idx)
        optimal = ACTION_NAMES[OPTIMAL_ACTIONS[class_idx]]
        is_correct = action_idx == OPTIMAL_ACTIONS[class_idx]

        st.markdown(f"""
        <div class="action-card {'action-correct' if is_correct else 'action-wrong'}">
            <div style="font-size: 2rem;">{'🛡️' if is_correct else '⚠️'}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {'#00d4aa' if is_correct else '#ff4757'};">
                {action_name}
            </div>
            <div style="color: #8892b0; margin-top: 8px;">
                For attack: <b>{selected_class}</b> | 
                Optimal: <b>{optimal}</b> |
                {'Match ✅' if is_correct else 'Mismatch ❌'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Save model
    st.markdown("---")
    if st.button("💾 Save DQN Model"):
        save_path = os.path.join(PROJECT_ROOT, "saved_models", "dqn_model")
        agent.save(save_path)
        st.success(f"✅ DQN model saved to `saved_models/{st.session_state.get("dataset_type", "nsl-kdd")}/dqn_model.zip`")

else:
    st.info("👆 Click **Train DQN Agent** to start reinforcement learning.")
