import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def create_counterfactual_dataset(x, x_demo, original_treatment, mask, device):
    """
    Creates a dataset with the original data but ready for counterfactual simulation
    
    Args:
        x: Patient temporal features [batch_size, seq_len, features]
        x_demo: Patient static features [batch_size, demo_features]
        original_treatment: Original treatment sequence [batch_size, seq_len]
        mask: Mask for valid timesteps [batch_size, seq_len]
        device: torch device
    """
    # Convert to tensors if not already
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(x_demo):
        x_demo = torch.tensor(x_demo, dtype=torch.float32)
    if not torch.is_tensor(original_treatment):
        original_treatment = torch.tensor(original_treatment, dtype=torch.float32)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, dtype=torch.float32)
        
    # Move to device
    x = x.to(device)
    x_demo = x_demo.to(device)
    original_treatment = original_treatment.to(device)
    mask = mask.to(device)
    
    # Create dummy y and death tensors (not used in inference)
    dummy_y = torch.zeros_like(original_treatment)
    dummy_death = torch.zeros(len(x), dtype=torch.float32).to(device)
    
    dataset = TensorDataset(x, x_demo, original_treatment, dummy_y, dummy_y, dummy_death, mask)
    return dataset

def simulate_counterfactual(model, x, x_demo, original_treatment, new_treatment, mask, device, batch_size=32):
    """
    Simulates counterfactual trajectories given new treatment sequences
    
    Args:
        model: Trained Seq2Seq model
        x: Patient temporal features [batch_size, seq_len, features]
        x_demo: Patient static features [batch_size, demo_features]
        original_treatment: Original treatment sequence [batch_size, seq_len]
        new_treatment: New treatment sequence to simulate [batch_size, seq_len]
        mask: Mask for valid timesteps [batch_size, seq_len]
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        dict containing:
        - factual_trajectories: Predicted trajectories under original treatment
        - counterfactual_trajectories: Predicted trajectories under new treatment
        - treatment_effects: Difference between counterfactual and factual
        - uncertainty: Uncertainty estimates for the predictions
    """
    # Create dataset and dataloader
    dataset = create_counterfactual_dataset(x, x_demo, original_treatment, mask, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    factual_trajectories = []
    counterfactual_trajectories = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_batch, x_demo_batch, treatment_batch, _, y_batch, death_batch, mask_batch = batch
            
            # Get sequence length from input
            seq_len = x_batch.size(1)
            pred_len = y_batch.size(1)
            total_len = seq_len + pred_len  # Total length needed for treatment sequence
            
            # Ensure treatment sequences have enough timesteps for prediction
            treatment_batch = treatment_batch[:, :total_len]
            if treatment_batch.size(1) < total_len:
                # Pad with last treatment if sequence is too short
                pad_size = total_len - treatment_batch.size(1)
                last_treatment = treatment_batch[:, -1:].repeat(1, pad_size)
                treatment_batch = torch.cat([treatment_batch, last_treatment], dim=1)
            
            # Get predictions for original treatment
            output_factual, _, patient_rep, _, _ = model(
                x_batch, y_batch, x_demo_batch, treatment_batch, 
                teacher_forcing_ratio=0
            )
            
            # Get predictions for new treatment
            new_treatment_batch = new_treatment[len(factual_trajectories):len(factual_trajectories) + len(x_batch)]
            new_treatment_batch = torch.tensor(new_treatment_batch, dtype=torch.float32).to(device)
            
            # Apply same truncation/padding to new treatment
            new_treatment_batch = new_treatment_batch[:, :total_len]
            if new_treatment_batch.size(1) < total_len:
                pad_size = total_len - new_treatment_batch.size(1)
                last_treatment = new_treatment_batch[:, -1:].repeat(1, pad_size)
                new_treatment_batch = torch.cat([new_treatment_batch, last_treatment], dim=1)
            
            _, output_counterfactual, _, _, _ = model(
                x_batch, y_batch, x_demo_batch, new_treatment_batch,
                teacher_forcing_ratio=0
            )
            
            factual_trajectories.append(output_factual.cpu().numpy())
            counterfactual_trajectories.append(output_counterfactual.cpu().numpy())
    
    # Combine batches
    factual_trajectories = np.concatenate(factual_trajectories, axis=0)
    counterfactual_trajectories = np.concatenate(counterfactual_trajectories, axis=0)
    
    # Calculate treatment effects
    treatment_effects = counterfactual_trajectories - factual_trajectories
    
    # Calculate uncertainty (per patient)
    if treatment_effects.ndim == 2:
        # If we have [patients, timesteps]
        uncertainty = np.std(treatment_effects, axis=1)  # One value per patient
    elif treatment_effects.ndim == 3:
        # If we have [patients, timesteps, features]
        uncertainty = np.std(treatment_effects, axis=1)  # One value per patient per feature
    else:
        uncertainty = np.zeros(len(treatment_effects))  # Fallback
    
    return {
        'factual_trajectories': factual_trajectories,
        'counterfactual_trajectories': counterfactual_trajectories,
        'treatment_effects': treatment_effects,
        'uncertainty': uncertainty
    }

def get_optimal_treatment(model, x, x_demo, mask, device, possible_treatments=None, batch_size=32):
    """
    Finds the optimal treatment sequence by simulating multiple possibilities
    
    Args:
        model: Trained Seq2Seq model
        x: Patient temporal features
        x_demo: Patient static features
        mask: Mask for valid timesteps
        device: torch device
        possible_treatments: List of treatment sequences to try (if None, generates binary combinations)
        batch_size: Batch size for processing
    
    Returns:
        dict containing:
        - optimal_treatment: Treatment sequence with best predicted outcome
        - optimal_trajectory: Predicted trajectory under optimal treatment
        - all_trajectories: Dictionary of all simulated trajectories
    """
    if possible_treatments is None:
        # Generate all possible binary treatment sequences for the prediction window
        seq_len = x.shape[1]
        num_combinations = 2 ** seq_len
        possible_treatments = np.array([
            [int(b) for b in format(i, f'0{seq_len}b')]
            for i in range(num_combinations)
        ])
    
    best_outcome = float('-inf')
    optimal_treatment = None
    optimal_trajectory = None
    all_trajectories = {}
    
    # Try each possible treatment sequence
    for treatment_seq in possible_treatments:
        treatment_seq = np.tile(treatment_seq, (len(x), 1))  # Repeat for batch
        results = simulate_counterfactual(
            model, x, x_demo, np.zeros_like(treatment_seq),  # Original treatment doesn't matter here
            treatment_seq, mask, device, batch_size
        )
        
        # Evaluate outcome (can be customized based on specific metrics)
        outcome = results['counterfactual_trajectories'].mean()
        all_trajectories[str(treatment_seq[0].tolist())] = {
            'treatment': treatment_seq[0],
            'trajectory': results['counterfactual_trajectories'][0],
            'outcome': outcome
        }
        
        if outcome > best_outcome:
            best_outcome = outcome
            optimal_treatment = treatment_seq[0]
            optimal_trajectory = results['counterfactual_trajectories'][0]
    
    return {
        'optimal_treatment': optimal_treatment,
        'optimal_trajectory': optimal_trajectory,
        'all_trajectories': all_trajectories
    } 