# Running T4 on Windows

This document describes the modifications made to run the T4 code on Windows. The original project was designed for Ubuntu 16.04 and with the addition of the MIMIC-III and/or AmsterdamUMCdb datasets, which require authentication to use, but with these modifications, it can run on Windows with synthetic data. We also add synthetic data from the [causal_modeling project](https://github.com/Blood-Glucose-Control/causal_modeling/tree/main/synthetic_data).

## Original Requirements
The original project required:
- Ubuntu 16.04
- Python 3.6
- PyTorch 1.4
- MIMIC-III dataset or synthetic data

## Training Process
The training happens in two phases:
1. **Pretraining Phase**: A single pretraining run with a fixed seed (default: 66)
2. **Main Training Phase**: Multiple training runs with different seeds (default: seeds 101-110)

Each training run:
- Runs for 50 epochs (configurable)
- Uses batch size of 32 (configurable)
- Tests on both MIMIC-III and AmsterdamDB datasets (using synthetic data as stand-in)
- Saves model checkpoints and logs

## Detailed Pipeline Description

### 1. Data Generation
Before training starts:
- Synthetic data is generated using `simulation/gen_synthetic.py`
- Creates `data/synthetic_full.pkl` with simulated patient data
- This replaces the need for real MIMIC-III/AmsterdamDB data

### 2. Pretraining Phase
First, the model goes through pretraining:
- Uses fixed seed (66) for reproducibility
- Runs for 50 epochs
- Creates a pretrained model: `checkpoints/[DATE]/[TIME]_[TAU]_pretrain.pt`
- Logs are saved to: `log/[DATE]/[TIME]_[TAU]_pretrain.log`
- This pretrained model will be used as starting point for all subsequent training runs

### 3. Main Training Phase
For each seed (101 through 110):
1. **Model Initialization**:
   - Loads the pretrained model
   - Initializes with the current seed
   - Sets up new log files and checkpoint paths

2. **Training**:
   - Runs for 50 epochs
   - Each epoch:
     - Trains on synthetic data
     - Validates performance
     - Saves if best model so far
   - Saves final model as: `checkpoints/[DATE]/[TIME]_[TAU]_0.4_[SEED].pt`
   - Logs to: `log/[DATE]/[TIME]_[TAU]_0.4_[SEED].log`

3. **Testing**:
   - Tests on synthetic data (standing in for MIMIC-III)
   - Tests on synthetic data (standing in for AmsterdamDB)
   - Tests for both 30-day and 60-day mortality windows
   - Reports:
     - Difference rates
     - Same rates
     - Number of differences
     - Total rates
     - Number of patients

4. **Results**:
   - Saved in `results_mimic/[DATE]/[TIME]_[TAU]_0.4/`
   - Includes performance metrics for both datasets

### Output Directory Structure
```
project/
├── checkpoints/[DATE]/          # Model checkpoints
│   ├── [TIME]_[TAU]_pretrain.pt    # Pretrained model
│   ├── [TIME]_[TAU]_0.4_101.pt     # Model for seed 101
│   └── ...                          # Models for other seeds
├── log/[DATE]/                  # Training logs
│   ├── [TIME]_[TAU]_pretrain.log   # Pretraining logs
│   ├── [TIME]_[TAU]_0.4_101.log    # Logs for seed 101
│   └── ...                          # Logs for other seeds
└── results_mimic/[DATE]/        # Final results
    └── [TIME]_[TAU]_0.4/           # Results for all seeds
```

Where:
- `[DATE]`: Current date (YYYY-MMDD)
- `[TIME]`: Current time (HHMMSS)
- `[TAU]`: Follow-up steps parameter (e.g., 3)
- `0.4`: Fixed augmentation ratio
- `101-110`: Different random seeds

### Configuring Training Parameters
All main training parameters can be found at the top of `run.ps1`:
```powershell
$epochs = 50              # Number of epochs per training run
$batch_size = 32         # Batch size for training
$lr = "5e-5"            # Learning rate
$seed = 66              # Seed for pretraining phase
```

### Configuring Number of Seeds
The number of different seeds to test is controlled by this line in `run.ps1`:
```powershell
101..110 | ForEach-Object {  # Tests seeds 101 through 110
    $current_seed = $_
    # ... training code ...
}
```
To change the number of seeds:
- Modify the range (e.g., `101..120` for 20 seeds)
- The first number is the starting seed
- The second number is the ending seed
- Total runs = (ending seed - starting seed + 1)

## Changes Made

### 1. Script Conversion
Converted the bash script (`run.sh`) to PowerShell (`run.ps1`) for Windows compatibility:
```powershell
# PowerShell equivalent of run.sh
$env:CUDA_VISIBLE_DEVICES = 1
$TAU = $args[0]
$date = Get-Date -Format "yyyy-MMdd"
$time = Get-Date -Format "HHmmss"
# ... rest of the script
```

### 2. Python Environment Setup
Created a dedicated virtual environment with compatible package versions:
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies with specific versions
pip install "numpy<2.0"  # Required for compatibility
pip install torch
pip install scikit-learn
```

### 3. Data Handling
Modified `dataset.py` to handle synthetic data format. The synthetic data has a different structure than the expected MIMIC-III format:

Original synthetic data format:
```python
{
    'x': temporal_features,
    'x_static': static_features,
    'y': outcomes,
    'a': treatments
}
```

Expected format:
```python
{
    'feature_N': [...],  # temporal features
    'outcome': [...],
    'treatment': [...],
    'death': value,
    'agegroup': value,
    'heightgroup': value,
    'weightgroup': value,
    'gender': value
}
```

Added conversion logic in `dataset.py` to transform between these formats.

### 4. File Path Handling
Updated Python script paths in `run.ps1` to use correct directory structure:
```powershell
python model/pre_train.py  # Instead of python pre_train.py
python model/main.py       # Instead of python main.py
```

### 5. Argument Parsing
Fixed argument parsing in PowerShell script using array syntax:
```powershell
$pre_train_args = @(
    "--epochs", $epochs,
    "--batch_size", $batch_size,
    # ... other arguments
)
python model/pre_train.py $pre_train_args
```

## Usage

1. Clone the repository
2. Set up the Python environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   pip install "numpy<2.0"
   pip install torch scikit-learn
   ```

3. Generate synthetic data:
   ```powershell
   python simulation/gen_synthetic.py
   ```

4. Run the training script:
   ```powershell
   .\run.ps1 3  # Where 3 is the number of follow-up steps
   ```

## Known Limitations
- Currently uses synthetic data only (MIMIC-III dataset support would require additional modifications)
- Death prediction is set to 0 for synthetic data
- Demographic features are approximated from static features

## Original Project
This is a modified version of the T4 project for Windows compatibility. For the original project and research paper, please refer to the [original repository](https://github.com/ruoqi-liu/T4/tree/main). 