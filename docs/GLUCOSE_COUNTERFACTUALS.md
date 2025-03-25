# Blood Glucose Counterfactual Analysis with T4

This document describes the modifications made to the T4 model to enable counterfactual analysis of blood glucose levels based on insulin dosage and timing.

## Overview

The original T4 model has been extended to:

1. Handle continuous insulin dosage values (0-1 normalized)
2. Incorporate insulin timing as a new treatment dimension
3. Predict blood glucose trajectories based on these interventions
4. Show the counterfactual evolution of blood glucose for 3-5 hours after intervention

## Key Modifications

### 1. Model Architecture Changes

#### Treatment Representation
- Modified treatment vector to have 2 dimensions:
  - `treatment[:, 0]`: Insulin dosage (0-1 normalized value)
  - `treatment[:, 1]`: Insulin timing (0-1 normalized where 0.5 = exact timing, <0.5 = early, >0.5 = late)

#### Encoder
- Updated RNN to accept 2D treatment input
- Modified propensity score prediction to output two values (dosage and timing)
- Enhanced state tracking to maintain both treatment dimensions

#### Decoder
- Added specialized output branches:
  - `fc_out_dose_high/low`: For high/low insulin dose effects
  - `fc_out_timing_early/late`: For early/late insulin timing effects
- Combined predictions from both dimensions with appropriate weighting

### 2. Data Processing

#### Glucose Data Conversion
- Created `convert_glucose_data_to_t4_format()` to prepare glucose data for the model
- Normalized insulin doses to 0-1 range for model compatibility
- Converted timing information to normalized 0-1 scale

#### Treatment Sequence Handling
- Updated `pad_and_truncate()` to handle 2D treatment vectors
- Implemented new binning strategy for treatment classification

### 3. Counterfactual Generation

#### Main Functions
- `generate_glucose_counterfactuals()`: Creates counterfactual scenarios by modifying:
  - Insulin dosage (e.g., 50% more or less insulin)
  - Insulin timing (e.g., 15 minutes earlier or later)
- Handles both individual and combined modifications

#### Visualization
- `plot_glucose_counterfactuals()`: Creates detailed visualizations showing:
  - Historical and predicted glucose trajectories
  - Insulin doses and meal carbs
  - Effect sizes and clinical implications

### 4. Example Usage

```python
# 1. Load model
model = load_model('checkpoints/best_model.pt')

# 2. Generate or load glucose data
data_window = # DataFrame with glucose, insulin, carbs, etc.

# 3. Select intervention point
intervention_time = '2024-01-01 12:30:00'

# 4. Generate counterfactual for modified insulin dose
results = generate_glucose_counterfactuals(
    model=model,
    data_window=data_window,
    intervention_time=intervention_time,
    dose_multiplier=0.75,  # Reduce insulin by 25%
    timing_shift=None      # Keep original timing
)

# 5. Generate counterfactual for modified timing
results = generate_glucose_counterfactuals(
    model=model,
    data_window=data_window,
    intervention_time=intervention_time,
    dose_multiplier=None,  # Keep original dose
    timing_shift=-15       # Give insulin 15 minutes earlier
)

# 6. Visualize the results
fig = plot_glucose_counterfactuals(data_window, results)
```

## Clinical Relevance

The counterfactual analysis predicts how blood glucose levels would evolve under different insulin dosage and timing scenarios, which can help:

1. **Optimize Insulin Dosing**: Determine the right amount of insulin for specific meals
2. **Improve Timing**: Find the optimal pre-meal timing for insulin administration
3. **Prevent Hypoglycemia**: Identify risky dosing combinations
4. **Personalized Recommendations**: Tailor insulin regimens to individual needs

## Training Data Requirements

To train this model effectively, you need glucose data with:

1. Continuous glucose measurements (e.g., from CGM devices)
2. Insulin dosage records with timestamps
3. Meal carbohydrate information
4. Optional: Additional factors like exercise, stress levels

## Limitations

1. **Short-Term Predictions**: The model focuses on 3-5 hour windows, as glucose levels typically converge with original trajectories after this time
2. **Simplified Physiology**: The model may not capture all complex interactions in glucose metabolism
3. **Data Quality Dependence**: Predictions are only as good as the training data

## Future Enhancements

1. Add support for multiple insulin types (rapid/long-acting)
2. Incorporate exercise counterfactuals
3. Add meal composition effects beyond just carbohydrates
4. Implement uncertainty estimation in predictions

## Demo Script

A demonstration script is available at `examples/glucose_counterfactual_demo.py` showing how to:
1. Generate synthetic glucose data
2. Run dose and timing counterfactual analyses
3. Visualize and compare different counterfactual scenarios 