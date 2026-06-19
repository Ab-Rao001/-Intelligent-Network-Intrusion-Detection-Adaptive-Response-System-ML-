import os
import argparse
import numpy as np

from preprocessing.preprocessor import DataPreprocessor
from clustering.cluster_model import ClusteringPipeline
from classification.ann_model import ANNClassifier
from reinforcement_learning.dqn_agent import NetworkDefenseEnv, DQNAgent

def train_pipeline(dataset_path, dataset_type="nsl-kdd", n_samples=None):
    save_dir = f"saved_models/{dataset_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training pipeline on {dataset_path} ({dataset_type})...")
    
    # 1. Preprocessing
    print("\n[1/4] Preprocessing Data...")
    pp = DataPreprocessor()
    X_train, X_test, y_train, y_test, feats, df_clean = pp.run_pipeline(
        filepath=dataset_path, 
        dataset_type=dataset_type,
        n_samples=n_samples
    )
    pp.save(directory=save_dir)
    print(f"Data Preprocessed: Train {X_train.shape}, Test {X_test.shape}")
    
    # 2. Clustering
    print("\n[2/4] Training Clustering Models (PCA + K-Means)...")
    cluster_pipeline = ClusteringPipeline(n_pca_components=3, n_clusters=5)
    pca_data, cluster_labels, silhouette = cluster_pipeline.run(X_train)
    cluster_pipeline.save(directory=save_dir)
    print(f"Clustering completed. Silhouette Score: {silhouette:.4f}")
    
    # 3. Classification (ANN)
    print("\n[3/4] Training ANN Classifier...")
    ann = ANNClassifier(input_dim=X_train.shape[1])
    ann.train(X_train, y_train, X_test, y_test, epochs=10, save_dir=save_dir)
    ann_results = ann.evaluate(X_test, y_test)
    ann.save(path=os.path.join(save_dir, "ann_model.keras"))
    print(f"ANN Training completed. Accuracy: {ann_results['accuracy']:.4f}")
    
    # 4. Reinforcement Learning (DQN)
    print("\n[4/4] Training DQN Agent for Adaptive Response...")
    # Get class distribution from the data to train the RL agent on realistic probabilities
    class_counts = np.bincount(y_train, minlength=5)
    attack_distribution = class_counts / class_counts.sum()
    
    env = NetworkDefenseEnv(attack_distribution=attack_distribution, max_steps=200)
    agent = DQNAgent(env, verbose=0)
    agent.train(total_timesteps=10000)
    agent.save(path=os.path.join(save_dir, "dqn_model"))
    eval_results = agent.evaluate(n_episodes=5)
    print(f"DQN Training completed. Agent Correct Rate: {eval_results['avg_correct_rate']:.2%}")
    
    print("\nFull Pipeline Execution Finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models for the Network IDS.")
    parser.add_argument("--dataset", type=str, default="dataset/archive/KDDTrain+.txt", help="Path to the dataset CSV/TXT")
    parser.add_argument("--type", type=str, default="nsl-kdd", choices=["nsl-kdd", "cicids"], help="Dataset type")
    parser.add_argument("--samples", type=int, default=None, help="Number of rows to sample (useful for large datasets)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
    else:
        train_pipeline(args.dataset, args.type, args.samples)
