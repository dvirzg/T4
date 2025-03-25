#!/usr/bin/env python
# Glucose counterfactual demo script

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simple_glucose_gen import EnhancedGlucoseGenerator, plot_glucose_data
from counterfactuals.counterfactual_testing import (
    convert_glucose_data_to_t4_format,
    generate_glucose_counterfactuals,
    plot_glucose_counterfactuals
)
from model.seq2seq import Encoder, AttentionDecoder, Seq2Seq

def load_model(checkpoint_path, device='cuda'):
    """Load a trained T4 model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['args']
    
    # Initialize model components
    encoder = Encoder(
        input_dim=model_args.vital_num,
        output_dim=1,
        x_static_size=model_args.demo_dim,
        emb_dim=model_args.emb_dim,
        hid_dim=model_args.hidden_dim,
        n_layers=model_args.layer_num,
        dropout=model_args.dropout,
        device=device
    )
    
    decoder = AttentionDecoder(
        output_dim=1,
        x_static_size=model_args.demo_dim,
        emb_dim=model_args.emb_dim,
        hid_dim=model_args.hidden_dim,
        n_layers=model_args.layer_num,
        dropout=model_args.dropout
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    return model

def generate_sample_data(days=3, seed=42):
    """Generate sample glucose data"""
    generator = EnhancedGlucoseGenerator(seed=seed)
    df = generator.generate_data(days=days)
    
    # Plot the generated data
    fig = plot_glucose_data(df)
    plt.tight_layout()
    plt.savefig('sample_glucose_data.png')
    plt.close()
    
    print(f"Generated {days} days of sample glucose data")
    return df

def run_dose_counterfactual_analysis(model, data, device='cuda'):
    """Run counterfactual analysis for different insulin doses"""
    # Find a suitable intervention point (insulin dose > 5)
    insulin_events = data[data['insulin'] > 5]
    if len(insulin_events) == 0:
        print("No suitable insulin events found")
        return
    
    # Choose a random intervention time
    intervention_time = insulin_events.index[np.random.randint(0, len(insulin_events))]
    print(f"Selected intervention time: {intervention_time}")
    
    # Create window around intervention
    window_start = intervention_time - timedelta(hours=2)
    window_end = intervention_time + timedelta(hours=5)
    data_window = data.loc[window_start:window_end].copy()
    
    # Run counterfactuals with different dose multipliers
    dose_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    
    for multiplier in dose_multipliers:
        print(f"Generating counterfactual with dose multiplier: {multiplier}")
        cf_result = generate_glucose_counterfactuals(
            model=model,
            data_window=data_window,
            intervention_time=intervention_time,
            dose_multiplier=multiplier,
            timing_shift=None,
            device=device
        )
        results.append((multiplier, cf_result))
    
    # Plot all results in a single figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot historical data
    historical_data = data_window.loc[:intervention_time]
    ax.plot(historical_data.index, historical_data['glucose'], 'b-', label='Historical', linewidth=2)
    
    # Plot factual prediction
    factual_result = results[2][1]  # multiplier = 1.0
    factual_pred = factual_result['factual_predictions']
    ax.plot(factual_pred.index, factual_pred['glucose'], 'b--', label='Factual (No Change)', linewidth=2)
    
    # Plot counterfactuals with different colors
    colors = ['red', 'orange', 'green', 'purple']
    for i, (multiplier, result) in enumerate([r for r in results if r[0] != 1.0]):
        cf_pred = result['counterfactual_predictions']
        ax.plot(cf_pred.index, cf_pred['glucose'], f'{colors[i]}--', 
                label=f'Dose × {multiplier}', linewidth=2)
    
    # Mark intervention point
    ax.axvline(x=intervention_time, color='black', linestyle='--')
    ax.annotate('Intervention', 
               xy=(intervention_time, data_window.loc[intervention_time, 'glucose']),
               xytext=(15, 15),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=10)
    
    # Format plot
    ax.set_title('Effect of Insulin Dose on Blood Glucose', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add target range
    ax.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
    
    plt.tight_layout()
    plt.savefig('dose_counterfactuals.png')
    plt.close()
    
    # Also create individual plots for each counterfactual
    for multiplier, result in results:
        fig = plot_glucose_counterfactuals(data_window, result)
        plt.savefig(f'counterfactual_dose_{multiplier:.2f}.png')
        plt.close()
    
    return results

def run_timing_counterfactual_analysis(model, data, device='cuda'):
    """Run counterfactual analysis for different insulin timing"""
    # Find a suitable intervention point (insulin dose > 5)
    insulin_events = data[data['insulin'] > 5]
    if len(insulin_events) == 0:
        print("No suitable insulin events found")
        return
    
    # Choose a random intervention time
    intervention_time = insulin_events.index[np.random.randint(0, len(insulin_events))]
    print(f"Selected intervention time: {intervention_time}")
    
    # Create window around intervention
    window_start = intervention_time - timedelta(hours=2)
    window_end = intervention_time + timedelta(hours=5)
    data_window = data.loc[window_start:window_end].copy()
    
    # Run counterfactuals with different timing shifts
    timing_shifts = [-30, -15, 0, 15, 30]  # minutes
    results = []
    
    for shift in timing_shifts:
        print(f"Generating counterfactual with timing shift: {shift} minutes")
        cf_result = generate_glucose_counterfactuals(
            model=model,
            data_window=data_window,
            intervention_time=intervention_time,
            dose_multiplier=None,
            timing_shift=shift,
            device=device
        )
        results.append((shift, cf_result))
    
    # Plot all results in a single figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot historical data
    historical_data = data_window.loc[:intervention_time]
    ax.plot(historical_data.index, historical_data['glucose'], 'b-', label='Historical', linewidth=2)
    
    # Plot factual prediction
    factual_result = results[2][1]  # shift = 0
    factual_pred = factual_result['factual_predictions']
    ax.plot(factual_pred.index, factual_pred['glucose'], 'b--', label='Factual (No Change)', linewidth=2)
    
    # Plot counterfactuals with different colors
    colors = ['red', 'orange', 'green', 'purple']
    for i, (shift, result) in enumerate([r for r in results if r[0] != 0]):
        cf_pred = result['counterfactual_predictions']
        label = f'Earlier {abs(shift)}m' if shift < 0 else f'Later {shift}m'
        ax.plot(cf_pred.index, cf_pred['glucose'], f'{colors[i]}--', 
                label=label, linewidth=2)
    
    # Mark intervention point
    ax.axvline(x=intervention_time, color='black', linestyle='--')
    ax.annotate('Intervention', 
               xy=(intervention_time, data_window.loc[intervention_time, 'glucose']),
               xytext=(15, 15),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=10)
    
    # Format plot
    ax.set_title('Effect of Insulin Timing on Blood Glucose', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add target range
    ax.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
    
    plt.tight_layout()
    plt.savefig('timing_counterfactuals.png')
    plt.close()
    
    # Also create individual plots for each counterfactual
    for shift, result in results:
        fig = plot_glucose_counterfactuals(data_window, result)
        plt.savefig(f'counterfactual_timing_{shift:+d}.png')
        plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='T4 Glucose Counterfactual Demo')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run model on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.chdir('output')
    
    # Generate sample data
    data = generate_sample_data(days=5, seed=args.seed)
    
    # Load model
    try:
        model = load_model(args.checkpoint, device=args.device)
    except FileNotFoundError:
        print(f"Model checkpoint not found at {args.checkpoint}")
        print("Using a randomly initialized model for demonstration purposes.")
        # Create a dummy model with appropriate dimensions
        encoder = Encoder(
            input_dim=6,  # glucose, carbs, exercise, stress, active_insulin, carb_impact
            output_dim=1,
            x_static_size=4,  # agegroup, heightgroup, weightgroup, gender
            emb_dim=64,
            hid_dim=128,
            n_layers=2,
            dropout=0.1,
            device=args.device
        )
        
        decoder = AttentionDecoder(
            output_dim=1,
            x_static_size=4,
            emb_dim=64,
            hid_dim=128,
            n_layers=2,
            dropout=0.1
        )
        
        model = Seq2Seq(encoder, decoder, args.device).to(args.device)
    
    # Run counterfactual analyses
    dose_results = run_dose_counterfactual_analysis(model, data, device=args.device)
    timing_results = run_timing_counterfactual_analysis(model, data, device=args.device)
    
    print("Counterfactual analysis complete. Results saved to output directory.")

if __name__ == "__main__":
    main() 