import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_trajectories(factual_trajectory, counterfactual_trajectory, 
                     original_treatment, new_treatment, 
                     uncertainty=None, feature_names=None, 
                     title=None, save_path=None):
    """
    Plot factual vs counterfactual trajectories with treatment sequences
    
    Args:
        factual_trajectory: Array of factual outcomes [timesteps]
        counterfactual_trajectory: Array of counterfactual outcomes [timesteps]
        original_treatment: Original treatment sequence [timesteps]
        new_treatment: Counterfactual treatment sequence [timesteps]
        uncertainty: Optional uncertainty estimates [timesteps]
        feature_names: Optional list of feature names
        title: Optional title for the plot
        save_path: Optional path to save the figure
    """
    # Ensure all sequences have the same length
    timesteps = len(factual_trajectory)
    
    # Pad or truncate treatment sequences if needed
    if len(original_treatment) > timesteps:
        original_treatment = original_treatment[:timesteps]
    elif len(original_treatment) < timesteps:
        original_treatment = np.pad(original_treatment, 
                                  (0, timesteps - len(original_treatment)), 
                                  'edge')
    
    if len(new_treatment) > timesteps:
        new_treatment = new_treatment[:timesteps]
    elif len(new_treatment) < timesteps:
        new_treatment = np.pad(new_treatment, 
                             (0, timesteps - len(new_treatment)), 
                             'edge')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.tight_layout(pad=3.0)
    
    # Plot trajectories
    time = np.arange(timesteps)
    ax1.plot(time, factual_trajectory, 'b-', label='Factual', linewidth=2)
    ax1.plot(time, counterfactual_trajectory, 'r--', label='Counterfactual', linewidth=2)
    
    # Add uncertainty if provided
    if uncertainty is not None:
        if np.isscalar(uncertainty):
            # If uncertainty is a single value, use it for all timesteps
            uncertainty = np.full_like(counterfactual_trajectory, uncertainty)
        elif len(uncertainty) < timesteps:
            # If uncertainty array is too short, pad it
            uncertainty = np.pad(uncertainty, (0, timesteps - len(uncertainty)), 'edge')
        elif len(uncertainty) > timesteps:
            # If uncertainty array is too long, truncate it
            uncertainty = uncertainty[:timesteps]
            
        ax1.fill_between(time, 
                        counterfactual_trajectory - uncertainty,
                        counterfactual_trajectory + uncertainty,
                        color='r', alpha=0.2, label='Uncertainty')
    
    # Add labels and title
    if feature_names is not None and len(feature_names) == 1:
        ax1.set_ylabel(feature_names[0])
    else:
        ax1.set_ylabel('Outcome')
    ax1.set_xlabel('Time')
    if title:
        fig.suptitle(title, fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # Plot treatment sequences
    ax2.step(time, original_treatment, 'b-', label='Original Treatment', where='post', linewidth=2)
    ax2.step(time, new_treatment, 'r--', label='Counterfactual Treatment', where='post', linewidth=2)
    
    # Highlight treatment differences
    diff_mask = original_treatment != new_treatment
    diff_times = time[diff_mask]
    if len(diff_times) > 0:  # Only add vlines if there are differences
        ax2.vlines(diff_times, -0.1, 1.1, colors='gray', alpha=0.3, label='Treatment Change')
    
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel('Treatment')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True)
    
    # Add intervention description
    intervention_text = "Intervention:\n"
    changes = []
    for t in diff_times:
        changes.append(f"t={t}: {int(original_treatment[t])}→{int(new_treatment[t])}")
    intervention_text += "\n".join(changes) if changes else "No changes"
    
    # Add text box with intervention description
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(1.02, 0.5, intervention_text, transform=ax1.transAxes, 
             verticalalignment='center', bbox=props)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_multiple_trajectories(results_dict, feature_idx=0, num_patients=5, 
                             feature_names=None, save_dir=None):
    """
    Plot trajectories for multiple patients
    
    Args:
        results_dict: Dictionary containing simulation results
        feature_idx: Index of the feature to plot
        num_patients: Number of patients to plot
        feature_names: Optional list of feature names
        save_dir: Optional directory to save figures
    """
    factual = results_dict['factual_trajectories']
    counterfactual = results_dict['counterfactual_trajectories']
    effects = results_dict['treatment_effects']
    uncertainty = results_dict.get('uncertainty', None)
    
    num_patients = min(num_patients, len(factual))
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_patients):
        if feature_names is not None:
            fname = feature_names[feature_idx]
            title = f"Patient {i} - Feature: {fname}"
        else:
            title = f"Patient {i}"
            
        save_path = save_dir / f"patient_{i}.png" if save_dir else None
        
        # Get single feature trajectories if multi-dimensional
        f_traj = factual[i] if factual[i].ndim == 1 else factual[i, :, feature_idx]
        cf_traj = counterfactual[i] if counterfactual[i].ndim == 1 else counterfactual[i, :, feature_idx]
        
        # Get uncertainty for this patient if available
        if uncertainty is not None:
            if uncertainty.ndim == 1:
                # Single uncertainty value per patient
                unc = uncertainty[i]
            elif uncertainty.ndim == 2:
                # Uncertainty per timestep per patient
                unc = uncertainty[i]
            else:
                # Uncertainty per timestep per feature per patient
                unc = uncertainty[i, :, feature_idx]
        else:
            unc = None
            
        # Create treatment sequences based on sequence length
        t_length = len(f_traj)
        orig_treatment = np.zeros(t_length)  # Default to no treatment
        new_treatment = np.ones(t_length)    # Default to all treatment
        
        plot_trajectories(
            f_traj, cf_traj,
            orig_treatment, new_treatment,
            uncertainty=unc,
            feature_names=[fname] if feature_names else None,
            title=title,
            save_path=save_path
        ) 