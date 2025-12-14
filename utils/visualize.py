import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def visualize_pair(particle_type, pred_pos, gt_pos, metadata):
    """
    Creates a side-by-side video comparing Prediction vs Ground Truth.
    
    pred_pos: [time, particles, dim]
    gt_pos: [time, particles, dim]
    """
    
    # Bounds for the plot
    bounds = metadata["bounds"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Setup plots
    axes[0].set_title("Prediction")
    axes[1].set_title("Ground Truth")
    
    for ax in axes:
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_aspect('equal')

    # Color map for particles (Water, Sand, Goop, etc.)
    # 9 distinct colors for up to 9 particle types
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    # Create scatter objects
    # We need to filter by particle type to assign colors
    # For efficiency in animation, we update data rather than recreating plots
    
    scatters_pred = []
    scatters_gt = []
    
    # Helper to split indices by type
    unique_types = np.unique(particle_type)
    type_indices = {t: (particle_type == t).nonzero(as_tuple=True)[0] for t in unique_types}

    for t in unique_types:
        color = colors[t % len(colors)]
        # Prediction scatter
        scatters_pred.append(axes[0].scatter([], [], c=color, s=10))
        # GT scatter
        scatters_gt.append(axes[1].scatter([], [], c=color, s=10))

    def update(frame):
        for i, t in enumerate(unique_types):
            idx = type_indices[t]
            
            # Update Prediction
            pos_p = pred_pos[frame][idx]
            scatters_pred[i].set_offsets(pos_p.cpu().numpy())
            
            # Update Ground Truth
            # Note: GT might have a slightly different length or offset depending on windowing
            # usually GT is aligned, but ensure frame < gt_len
            if frame < len(gt_pos):
                pos_g = gt_pos[frame][idx]
                scatters_gt[i].set_offsets(pos_g.cpu().numpy())
        return scatters_pred + scatters_gt

    anim = animation.FuncAnimation(
        fig, update, frames=len(pred_pos), interval=20, blit=True
    )
    
    plt.close()
    return anim