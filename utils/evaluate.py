import torch
import os
import matplotlib.pyplot as plt
from dataset import RolloutDataset
from model import LearnedSimulator
from rollout import rollout
from visualize import visualize_pair

# Configuration
MODEL_PATH = "temp/models/WaterDrop/checkpoint_5000.pt" # Update with your actual checkpoint
DATA_PATH = "temp/datasets/WaterDrop"
OUTPUT_DIR = "temp/videos"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Dataset
    print("Loading test dataset...")
    # Using 'valid' or 'test' split
    ds = RolloutDataset(DATA_PATH, "valid") 

    # 2. Load Model
    print("Loading model...")
    # Determine params from dataset metadata or saved config
    # For now, using defaults from model.py
    simulator = LearnedSimulator()
    
    # Load weights
    # Note: If you saved inside a loop, ensure the path exists
    try:
        simulator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Checkpoint not found at {MODEL_PATH}. Please train the model first.")
        return

    simulator.to(device)

    # 3. Run Rollouts
    num_rollouts = 3
    print(f"Generating {num_rollouts} rollouts...")

    for i in range(num_rollouts):
        print(f"Processing rollout {i+1}/{num_rollouts}...")
        
        # Get a specific trajectory
        # Offset by some amount to get interesting cases if needed
        data = ds[i] 
        
        # Run simulation
        pred_pos = rollout(simulator, data, ds.metadata, noise_std=0.0)
        
        # Align Ground Truth (cut off the initial window used for warm-start)
        window_size = 6 # Default window size
        gt_pos = data["position"][window_size:].permute(1, 0, 2) # [Time, N, Dim]
        
        # pred_pos is [Time, N, Dim]
        
        # Render
        print("Rendering video...")
        anim = visualize_pair(
            data["particle_type"],
            pred_pos,
            gt_pos,
            ds.metadata
        )
        
        save_path = os.path.join(OUTPUT_DIR, f"rollout_{i}.mp4")
        anim.save(save_path, writer="ffmpeg", fps=30)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()