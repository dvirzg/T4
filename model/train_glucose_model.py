#!/usr/bin/env python
# Script to train the modified T4 model on synthetic glucose data

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import random

# Import from our modules
from simulation.simple_glucose_gen import EnhancedGlucoseGenerator
from model.seq2seq import Encoder, AttentionDecoder, Seq2Seq
from counterfactuals.counterfactual_testing import convert_glucose_data_to_t4_format

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_glucose.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_glucose")

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_training_data(days=30, seed=42):
    """Generate synthetic glucose data for training"""
    generator = EnhancedGlucoseGenerator(seed=seed)
    logger.info(f"Generating {days} days of synthetic glucose data")
    return generator.generate_data(days=days)

def prepare_sequences(data, seq_length=48, pred_horizon=12, stride=6):
    """
    Prepare sequences from time series data for sequence-to-sequence modeling
    
    Args:
        data: Pandas DataFrame with glucose data
        seq_length: Length of input sequence (in 5-minute intervals, default 4 hours)
        pred_horizon: Prediction horizon (in 5-minute intervals, default 1 hour)
        stride: Stride between sequences (in 5-minute intervals)
        
    Returns:
        List of input-output sequences
    """
    sequences = []
    
    # Convert to t4 format first
    t4_data = convert_glucose_data_to_t4_format(data)
    features = t4_data['x']
    treatments = t4_data['treatment']
    
    # Create sequences with sliding window
    for i in range(0, len(features) - seq_length - pred_horizon, stride):
        # Input sequence: features and treatments
        x_seq = features[i:i+seq_length]
        treatment_seq = treatments[i:i+seq_length+pred_horizon]  # Include treatments for prediction period
        
        # Output sequence: glucose values to predict
        y_seq = np.zeros(pred_horizon + 1)  # +1 because we include the last value of input sequence
        y_seq[0] = features[i+seq_length-1, 0]  # First value is the current glucose
        y_seq[1:] = features[i+seq_length:i+seq_length+pred_horizon, 0]  # Future glucose values
        
        # Static features (demographics)
        x_demo = np.array([
            t4_data['agegroup'],
            t4_data['heightgroup'],
            t4_data['weightgroup'],
            t4_data['gender']
        ])
        
        sequences.append({
            'x': x_seq,
            'treatment': treatment_seq,
            'y': y_seq,
            'x_demo': x_demo
        })

    logger.info(f"Created {len(sequences)} sequences")
    return sequences

def create_dataset(sequences, device):
    """Convert sequences to PyTorch datasets"""
    # Create tensors for model
    x_list, x_demo_list, treatment_list, y_list = [], [], [], []
    
    for seq in sequences:
        x_list.append(torch.tensor(seq['x'], dtype=torch.float32))
        x_demo_list.append(torch.tensor(seq['x_demo'], dtype=torch.float32))
        treatment_list.append(torch.tensor(seq['treatment'], dtype=torch.float32))
        y_list.append(torch.tensor(seq['y'], dtype=torch.float32))
    
    # Stack all tensors
    x_tensor = torch.stack(x_list).to(device)
    x_demo_tensor = torch.stack(x_demo_list).to(device)
    treatment_tensor = torch.stack(treatment_list).to(device)
    y_tensor = torch.stack(y_list).to(device)
    
    # Create mask (all 1s for now since we're not dealing with variable sequence lengths)
    mask_tensor = torch.ones_like(x_tensor[:, :, 0], dtype=torch.float32).to(device)
    
    # Create death tensor (all 0s since this isn't relevant for glucose data)
    death_tensor = torch.zeros(len(sequences), dtype=torch.float32).to(device)
    
    # Create treatment label (not used in this application, but kept for compatibility)
    treatment_label = torch.zeros(len(sequences), dtype=torch.long).to(device)
    
    # Create dataset
    dataset = TensorDataset(
        x_tensor, x_demo_tensor, treatment_tensor, treatment_label,
        y_tensor, death_tensor, mask_tensor
    )
    
    return dataset

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Train the model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_dict = None
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, x_demo, treatment, treatment_label, y, death, mask = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, outputs_cf, patient_rep, ps_outputs, _ = model(
                x, y, x_demo, treatment, teacher_forcing_ratio=0.5
            )
            
            # Calculate loss on factual outputs only (not counterfactual)
            loss = criterion(outputs.squeeze(-1), y[:, 1:])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x, x_demo, treatment, treatment_label, y, death, mask = batch
                
                # Forward pass
                outputs, outputs_cf, patient_rep, ps_outputs, _ = model(
                    x, y, x_demo, treatment, teacher_forcing_ratio=0
                )
                
                # Calculate loss
                loss = criterion(outputs.squeeze(-1), y[:, 1:])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            logger.info(f"Saved best model at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    return best_model_dict

def main():
    parser = argparse.ArgumentParser(description='Train T4 model on glucose data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--days', type=int, default=60, help='Days of synthetic data to generate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run model on')
    
    args = parser.parse_args()
    
    # Model hyperparameters for saving
    args.vital_num = 6  # glucose, carbs, exercise, stress, active_insulin, carb_impact 
    args.demo_dim = 4   # agegroup, heightgroup, weightgroup, gender
    args.emb_dim = 64
    args.hidden_dim = 128
    args.layer_num = 2
    args.dropout = 0.1
    args.max_stay = 48  # 4 hours of 5-minute readings
    args.pre_window = 12  # 1 hour prediction horizon
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Generate synthetic data
    data = generate_training_data(days=args.days, seed=args.seed)
    
    # Prepare sequences for training
    sequences = prepare_sequences(data, seq_length=args.max_stay, pred_horizon=args.pre_window)
    
    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(sequences))
    random.shuffle(sequences)  # Shuffle before splitting
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]
    
    # Create datasets
    device = torch.device(args.device)
    train_dataset = create_dataset(train_sequences, device)
    val_dataset = create_dataset(val_sequences, device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    encoder = Encoder(
        input_dim=args.vital_num,  # glucose, carbs, exercise, stress, active_insulin, carb_impact
        output_dim=1,  # glucose prediction
        x_static_size=args.demo_dim,  # agegroup, heightgroup, weightgroup, gender
        emb_dim=args.emb_dim,
        hid_dim=args.hidden_dim,
        n_layers=args.layer_num,
        dropout=args.dropout,
        device=device
    )
    
    decoder = AttentionDecoder(
        output_dim=1,  # glucose prediction
        x_static_size=args.demo_dim,
        emb_dim=args.emb_dim,
        hid_dim=args.hidden_dim,
        n_layers=args.layer_num,
        dropout=args.dropout
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Train model
    logger.info("Starting model training")
    best_model_dict = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Save the best model
    if best_model_dict:
        checkpoint_path = f"checkpoints/glucose_model_epochs{args.epochs}_seed{args.seed}.pt"
        best_model_dict['args'] = args
        torch.save(best_model_dict, checkpoint_path)
        logger.info(f"Best model saved to {checkpoint_path}")
    
    logger.info("Training completed")

if __name__ == "__main__":
    main() 