import os

pages_dir = r"c:\Users\abrao\OneDrive\Desktop\University\AI\CyberSecurity\streamlit_app\pages"
files = os.listdir(pages_dir)

for file in files:
    if not file.endswith(".py"):
        continue
    filepath = os.path.join(pages_dir, file)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple search and replace for os.path.join(PROJECT_ROOT, "saved_models")
    new_content = content.replace(
        'os.path.join(PROJECT_ROOT, "saved_models")', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"))'
    )
    # Also replace direct string "saved_models/..."
    new_content = new_content.replace(
        '"saved_models/kmeans.pkl"', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"), "kmeans.pkl")'
    )
    new_content = new_content.replace(
        '"saved_models/pca.pkl"', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"), "pca.pkl")'
    )
    new_content = new_content.replace(
        '"saved_models/scaler.pkl"', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"), "scaler.pkl")'
    )
    new_content = new_content.replace(
        '"saved_models/ann_model.keras"', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"), "ann_model.keras")'
    )
    new_content = new_content.replace(
        '"saved_models/dqn_model"', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"), "dqn_model")'
    )
    new_content = new_content.replace(
        '"saved_models/target_encoder.pkl"', 
        'os.path.join(PROJECT_ROOT, "saved_models", st.session_state.get("dataset_type", "nsl-kdd"), "target_encoder.pkl")'
    )
    new_content = new_content.replace(
        'Models saved to `saved_models/`', 
        'Models saved to `saved_models/{st.session_state.get("dataset_type", "nsl-kdd")}/`'
    )
    new_content = new_content.replace(
        'DQN model saved to `saved_models/dqn_model.zip`', 
        'DQN model saved to `saved_models/{st.session_state.get("dataset_type", "nsl-kdd")}/dqn_model.zip`'
    )

    if content != new_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {file}")
