#!/bin/bash
# Quick demonstration script for the T4 Glucose Counterfactual Model

# Set up environment
echo "Setting up environment..."
mkdir -p checkpoints counterfactual_results

# Define parameters for a quick demonstration
DAYS=3
EPOCHS=5
BATCH_SIZE=32
SCENARIOS=2
DEVICE="cpu"  # Use CPU for demo (change to "cuda" if GPU is available)

# Train the model (with a smaller dataset and fewer epochs for quick demonstration)
echo "Training model with small dataset ($DAYS days) for $EPOCHS epochs..."
python train_glucose_model.py --days $DAYS --epochs $EPOCHS --batch_size $BATCH_SIZE --device $DEVICE

# Get the checkpoint path
CHECKPOINT="checkpoints/glucose_model_epochs${EPOCHS}_seed42.pt"

# Evaluate the model with counterfactual scenarios
echo "Evaluating model with counterfactual scenarios..."
python evaluate_counterfactuals.py --checkpoint $CHECKPOINT --mode both --scenarios $SCENARIOS --device $DEVICE

# Show results
echo "Demo completed! Results are available in:"
echo "  - counterfactual_results/dose/"
echo "  - counterfactual_results/timing/"
echo ""
echo "You can view the combined visualizations with:"
echo "  - counterfactual_results/dose/scenario_1_combined.png"
echo "  - counterfactual_results/timing/scenario_1_combined.png"
echo ""
echo "For full training and evaluation, run:"
echo "  python train_glucose_model.py --days 60 --epochs 50"
echo "  python evaluate_counterfactuals.py --checkpoint checkpoints/glucose_model_epochs50_seed42.pt" 