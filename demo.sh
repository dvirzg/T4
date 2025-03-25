#!/bin/bash
# Quick demonstration script for the T4 Glucose Counterfactual Model

# Set up environment
echo "Setting up environment..."
mkdir -p checkpoints counterfactuals/results

# Define parameters for a quick demonstration
DAYS=3
EPOCHS=5
BATCH_SIZE=32
SCENARIOS=2
DEVICE="cpu"  # Use CPU for demo (change to "cuda" if GPU is available)

# Set Python path to include current directory
export PYTHONPATH=.

# Train the model (with a smaller dataset and fewer epochs for quick demonstration)
echo "Training model with small dataset ($DAYS days) for $EPOCHS epochs..."
python3 model/train_glucose_model.py --days $DAYS --epochs $EPOCHS --batch_size $BATCH_SIZE --device $DEVICE

# Get the checkpoint path
CHECKPOINT="checkpoints/glucose_model_epochs${EPOCHS}_seed42.pt"

# Evaluate the model with counterfactual scenarios
echo "Evaluating model with counterfactual scenarios..."
python3 model/evaluate_counterfactuals.py --checkpoint $CHECKPOINT --mode both --scenarios $SCENARIOS --device $DEVICE

# Show results
echo "Demo completed! Results are available in:"
echo "  - counterfactuals/results/dose/"
echo "  - counterfactuals/results/timing/"
echo ""
echo "You can view the combined visualizations with:"
echo "  - counterfactuals/results/dose/scenario_1_combined.png"
echo "  - counterfactuals/results/timing/scenario_1_combined.png"
echo ""
echo "For full training and evaluation, run:"
echo "  python3 model/train_glucose_model.py --days 60 --epochs 50"
echo "  python3 model/evaluate_counterfactuals.py --checkpoint checkpoints/glucose_model_epochs50_seed42.pt" 