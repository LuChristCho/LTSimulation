import torch
from dataset import preprocess

def rollout(model, data, metadata, noise_std):
    """
    Performs a rollout (simulation) starting from the initial state in 'data'.
    """
    device = next(model.parameters()).device
    model.eval()
    
    window_length = data["position"].size(1)
    # The 'data' from RolloutDataset contains the full ground truth trajectory.
    # We only slice the first few steps to initialize the model.
    # Typically we need (window_length - 1) past steps to compute velocity.
    
    # We'll maintain a buffer of the current window of positions
    # Shape: [n_particles, window_length, dim]
    current_positions = data["position"][:, :window_length].to(device)
    
    trajectory = [current_positions[:, -1]] # Store the latest position
    
    # Number of steps to simulate (total steps in ground truth - initial window)
    total_steps = data["position"].size(1) - window_length
    
    with torch.no_grad():
        for _ in range(total_steps):
            # 1. Construct the graph for the current step
            # Note: target_position is None during inference
            graph = preprocess(
                data["particle_type"].to(device),
                current_positions,
                target_position=None,
                metadata=metadata,
                noise_std=0.0 # No noise during evaluation
            )
            
            graph = graph.to(device)
            
            # 2. Predict acceleration
            pred_acc = model(graph)
            
            # 3. Denormalize acceleration
            pred_acc = pred_acc * torch.sqrt(torch.tensor(metadata["acc_std"], device=device)**2 + noise_std**2) + torch.tensor(metadata["acc_mean"], device=device)
            
            # 4. Integrate (Euler integration)
            # recent_position = current_positions[:, -1]
            # velocity = recent_position - current_positions[:, -2]
            
            # More precise integration using the window:
            recent_pos = current_positions[:, -1]
            prev_pos = current_positions[:, -2]
            velocity = recent_pos - prev_pos
            
            next_pos = recent_pos + velocity + pred_acc 
            
            trajectory.append(next_pos)
            
            # 5. Update window
            # Shift window: remove oldest, add new prediction
            next_pos_reshaped = next_pos.unsqueeze(1)
            current_positions = torch.cat((current_positions[:, 1:], next_pos_reshaped), dim=1)

    # Stack trajectory into [n_steps, n_particles, dim]
    return torch.stack(trajectory)