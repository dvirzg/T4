import torch
import numpy as np
from model.dataset import load_and_process_data
from model.seq2seq import Encoder, AttentionDecoder as Decoder, Seq2Seq
from model.counterfactual_sim import simulate_counterfactual, get_optimal_treatment
from model.visualization import plot_trajectories, plot_multiple_trajectories
import argparse
from pathlib import Path
import sys
import logging

# Add model directory to path to handle old checkpoint references
sys.path.append('model')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path, args, device):
    """Load the trained model from checkpoint"""
    # Initialize model components
    encoder = Encoder(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        x_static_size=args.x_static_size,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        device=device
    )
    decoder = Decoder(
        output_dim=args.output_dim,
        x_static_size=args.x_static_size,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # Create model
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Load checkpoint with weights_only=False and handle module path differences
    import torch.serialization
    torch.serialization.add_safe_globals([('seq2seq', 'Seq2Seq', Seq2Seq)])
    
    try:
        # First try loading as a state dict
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If not a state dict, try loading as direct model
            loaded_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(loaded_model.state_dict())
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Attempting direct model transfer...")
        loaded_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Transfer parameters manually if needed
        model.load_state_dict(loaded_model.state_dict())
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Simulate counterfactual trajectories')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory containing data')
    parser.add_argument('--dataset', type=str, default='synthetic_full', help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_patients', type=int, default=5, help='Number of patients to visualize')
    parser.add_argument('--output_dir', type=str, default='results/counterfactuals', help='Output directory')
    
    # Model parameters (should match training)
    parser.add_argument('--input_dim', type=int, default=20, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--x_static_size', type=int, default=4, help='Static feature size')
    parser.add_argument('--emb_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_stay', type=int, default=20, help='Maximum stay length')
    parser.add_argument('--pre_window', type=int, default=3, help='Prediction window size')
    parser.add_argument('--load_cache', action='store_true', help='Whether to load cached dataset')
    parser.add_argument('--negative_sample', action='store_true', help='Whether to use negative sampling')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args, device)
    model.eval()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    features, dataset = load_and_process_data(args, device, logger, args.dataset)
    
    # Get a batch of data
    x, x_demo, treatment, _, y, death, mask = next(iter(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)))
    
    # Print dimensions for debugging
    logger.info(f"Data dimensions:")
    logger.info(f"x shape: {x.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"treatment shape: {treatment.shape}")
    logger.info(f"mask shape: {mask.shape}")
    
    # Calculate total sequence length needed
    input_seq_len = x.shape[1]
    pred_seq_len = y.shape[1]
    total_seq_len = input_seq_len + pred_seq_len
    
    # Ensure we have enough timesteps for prediction
    if treatment.shape[1] < total_seq_len:
        logger.warning(f"Treatment sequence length ({treatment.shape[1]}) is shorter than required length ({total_seq_len})")
        logger.info("Padding treatment sequences with last treatment value")
        pad_size = total_seq_len - treatment.shape[1]
        last_treatment = treatment[:, -1:].repeat(1, pad_size)
        treatment = torch.cat([treatment, last_treatment], dim=1)
    elif treatment.shape[1] > total_seq_len:
        logger.info(f"Truncating treatment sequences from {treatment.shape[1]} to {total_seq_len}")
        treatment = treatment[:, :total_seq_len]
    
    # Example 1: Simulate opposite treatment
    logger.info("\nSimulating opposite treatment...")
    new_treatment = 1 - treatment.cpu().numpy()  # Flip all treatments
    results = simulate_counterfactual(
        model, x, x_demo, treatment, new_treatment, mask, 
        device, batch_size=args.batch_size
    )
    
    # Plot results for multiple patients
    logger.info("\nGenerating visualizations...")
    plot_multiple_trajectories(
        results,
        feature_idx=0,  # Plot first feature
        num_patients=args.num_patients,
        feature_names=features if features else None,
        save_dir=output_dir / 'opposite_treatment'
    )
    
    # Example 2: Find optimal treatment
    logger.info("\nFinding optimal treatment...")
    # For demonstration, let's try with just the first patient and a shorter sequence
    short_x = x[:1, :5]  # First patient, first 5 timesteps
    short_x_demo = x_demo[:1]
    short_mask = mask[:1, :5]
    
    results = get_optimal_treatment(
        model, short_x, short_x_demo, short_mask, 
        device, possible_treatments=None, batch_size=args.batch_size
    )
    
    # Plot optimal treatment results
    # Get the baseline trajectory (no treatment) and optimal trajectory
    baseline_traj = results['all_trajectories']['[0, 0, 0, 0, 0]']['trajectory']
    optimal_traj = results['optimal_trajectory']
    
    # Get the original treatment sequence for the first patient (truncated to match)
    orig_treatment = treatment[0, :5].cpu().numpy()
    optimal_treatment = results['optimal_treatment']
    
    # Ensure all sequences have matching lengths
    traj_len = len(baseline_traj)
    if len(optimal_treatment) > traj_len:
        optimal_treatment = optimal_treatment[:traj_len]
    elif len(optimal_treatment) < traj_len:
        optimal_treatment = np.pad(optimal_treatment, (0, traj_len - len(optimal_treatment)), 'edge')
    
    if len(orig_treatment) > traj_len:
        orig_treatment = orig_treatment[:traj_len]
    elif len(orig_treatment) < traj_len:
        orig_treatment = np.pad(orig_treatment, (0, traj_len - len(orig_treatment)), 'edge')
    
    plot_trajectories(
        baseline_traj, optimal_traj,
        orig_treatment, optimal_treatment,
        title="Optimal Treatment Comparison",
        save_path=output_dir / 'optimal_treatment.png'
    )
    
    # Save numerical results
    np.save(output_dir / 'optimal_treatment.npy', optimal_treatment)
    np.save(output_dir / 'optimal_trajectory.npy', optimal_traj)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Generated visualizations:")
    logger.info(f"1. Opposite treatment trajectories: {output_dir}/opposite_treatment/")
    logger.info(f"2. Optimal treatment comparison: {output_dir}/optimal_treatment.png")

if __name__ == '__main__':
    main() 