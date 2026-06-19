"""
dqn_agent.py — Reinforcement Learning Module

Provides:
  - NetworkDefenseEnv: Custom Gymnasium environment
  - DQN agent training via Stable-Baselines3
  - Optimal response recommendation for detected attacks

State:  Attack class (one-hot encoded, 5 dims)
Actions:
  0 = Ignore
  1 = Alert Administrator
  2 = Block IP
  3 = Limit Traffic
  4 = Increase Monitoring

Reward Structure:
  +20  correct response
  -5   false alarm (acting on Normal traffic)
  -10  wrong/insufficient response
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value
    return func

# Try importing SB3 — graceful fallback if not installed
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

ATTACK_CLASSES = ["Normal", "DOS", "Probe", "R2L", "U2R"]

ACTION_NAMES = {
    0: "Ignore",
    1: "Alert Administrator",
    2: "Block IP",
    3: "Limit Traffic",
    4: "Increase Monitoring",
}

# ──────────────────────────────────────────────
# Optimal action mapping (ground truth for reward)
# ──────────────────────────────────────────────
OPTIMAL_ACTIONS = {
    0: 0,  # Normal → Ignore
    1: 2,  # DOS → Block IP
    2: 4,  # Probe → Increase Monitoring
    3: 1,  # R2L → Alert Administrator
    4: 3,  # U2R → Limit Traffic
}


class NetworkDefenseEnv(gym.Env):
    """Custom Gym environment for network defense RL.

    At each step, the environment presents an attack class (state),
    and the agent must choose the best defensive action.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, attack_distribution: np.ndarray | None = None, max_steps: int = 200):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Distribution of attack classes to sample from
        if attack_distribution is not None:
            self.attack_probs = attack_distribution / attack_distribution.sum()
        else:
            # Default: realistic distribution
            self.attack_probs = np.array([0.40, 0.35, 0.15, 0.07, 0.03])

        self.max_steps = max_steps
        self.current_step = 0
        self.current_attack = 0
        self.episode_rewards: list[float] = []
        self.total_reward = 0.0

    def _get_obs(self) -> np.ndarray:
        """One-hot encode current attack class."""
        obs = np.zeros(5, dtype=np.float32)
        obs[self.current_attack] = 1.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.current_attack = self.np_random.choice(5, p=self.attack_probs)
        return self._get_obs(), {}

    def step(self, action: int):
        optimal = OPTIMAL_ACTIONS[self.current_attack]

        # Reward logic
        if action == optimal:
            reward = 20.0
        elif self.current_attack == 0 and action != 0:
            reward = -5.0  # false alarm
        else:
            reward = -10.0  # wrong response

        self.total_reward += reward
        self.episode_rewards.append(reward)
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        # Next state
        if not terminated:
            self.current_attack = self.np_random.choice(5, p=self.attack_probs)

        info = {
            "attack_class": ATTACK_CLASSES[self.current_attack],
            "action_taken": ACTION_NAMES[action],
            "optimal_action": ACTION_NAMES[optimal],
            "reward": reward,
            "cumulative_reward": self.total_reward,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Step {self.current_step}: "
              f"Attack={ATTACK_CLASSES[self.current_attack]}, "
              f"Cumulative Reward={self.total_reward:.1f}")


class DQNAgent:
    """Wrapper around Stable-Baselines3 DQN for training and inference."""

    def __init__(self, env: NetworkDefenseEnv | None = None, **dqn_kwargs):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: "
                "pip install stable-baselines3[extra]"
            )

        self.env = env or NetworkDefenseEnv()
        default_kwargs = {
            "policy": "MlpPolicy",
            "learning_rate": linear_schedule(1e-3),
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 64,
            "gamma": 0.99,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "verbose": 1,
        }
        default_kwargs.update(dqn_kwargs)
        self.model = DQN(env=self.env, **default_kwargs)

    def train(self, total_timesteps: int = 20000, progress_callback=None):
        """Train the DQN agent.
        
        Args:
            total_timesteps: Total number of steps to train.
            progress_callback: Optional callable func(step, total_steps)
        """
        callbacks = []
        if progress_callback is not None:
            class ProgressCallback(BaseCallback):
                def __init__(self, total, verbose=0):
                    super().__init__(verbose)
                    self.total = total

                def _on_step(self) -> bool:
                    if self.num_timesteps % 500 == 0:
                        progress_callback(self.num_timesteps, self.total)
                    return True
            callbacks.append(ProgressCallback(total_timesteps))

        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        return self

    def predict(self, attack_class_index: int) -> tuple[int, str]:
        """Given an attack class index, return the recommended action.

        Returns:
            (action_index, action_name)
        """
        obs = np.zeros(5, dtype=np.float32)
        obs[attack_class_index] = 1.0
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        return action, ACTION_NAMES[action]

    def predict_all(self) -> dict:
        """Get recommended actions for all attack classes."""
        results = {}
        for i, cls in enumerate(ATTACK_CLASSES):
            action_idx, action_name = self.predict(i)
            results[cls] = {
                "action_index": action_idx,
                "action_name": action_name,
                "optimal_action": ACTION_NAMES[OPTIMAL_ACTIONS[i]],
                "is_correct": action_idx == OPTIMAL_ACTIONS[i],
            }
        return results

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate agent over multiple episodes.

        Returns:
            dict with avg_reward, correct_rate, episode_rewards
        """
        episode_rewards = []
        correct_counts = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0
            ep_correct = 0
            ep_total = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(int(action))
                ep_reward += reward
                ep_total += 1
                if reward == 20.0:
                    ep_correct += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            correct_counts.append(ep_correct / max(ep_total, 1))

        return {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_correct_rate": np.mean(correct_counts),
            "episode_rewards": episode_rewards,
        }

    def save(self, path: str = "saved_models/dqn_model"):
        """Save the DQN model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"DQN model saved to {path}")

    def load(self, path: str = "saved_models/dqn_model"):
        """Load a saved DQN model."""
        self.model = DQN.load(path, env=self.env)


if __name__ == "__main__":
    print("🛡️ Training DQN Agent for Network Defense...")
    env = NetworkDefenseEnv()
    agent = DQNAgent(env)
    agent.train(total_timesteps=10000)

    print("\nEvaluation:")
    eval_results = agent.evaluate(n_episodes=5)
    print(f"  Avg Reward: {eval_results['avg_reward']:.1f}")
    print(f"  Correct Rate: {eval_results['avg_correct_rate']:.2%}")

    print("\nRecommended Actions:")
    for cls, info in agent.predict_all().items():
        status = "Match" if info["is_correct"] else "Mismatch"
        print(f"  {cls:>10s} → {info['action_name']:<25s} {status}")

    agent.save()
