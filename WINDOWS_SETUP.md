# Running T4 on Windows

This document describes the modifications made to run the T4 code on Windows. The original project was designed for Ubuntu 16.04 and with the addition of the MIMIC-III and/or AmsterdamUMCdb datasets, which require authentication to use, but with these modifications, it can run on Windows with synthetic data. We also add synthetic data from the [causal_modeling project](https://github.com/Blood-Glucose-Control/causal_modeling/tree/main/synthetic_data).

## Original Requirements
The original project required:
- Ubuntu 16.04
- Python 3.6
- PyTorch 1.4
- MIMIC-III dataset or synthetic data

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