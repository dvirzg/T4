import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from counterfactual_testing import load_test_scenarios, CounterfactualEvaluator

# Load the test scenarios
scenarios = load_test_scenarios()

# Define a simple dummy prediction function
# In a real-world scenario, this would be your ML model's prediction function
def dummy_model(input_data, intervention_time, dose_multiplier, timing_shift):
    """
    A dummy model that naively adjusts the original glucose curve based on 
    the dose multiplier and timing shift.
    
    This is just a placeholder for demonstration - a real ML model would 
    be much more sophisticated.
    """
    # Get the original glucose values after intervention
    original_glucose = input_data.loc[intervention_time:, 'glucose'].values.astype(float)
    
    # Make a simple adjustment based on dose multiplier
    # This is a very naive approach just for demonstration
    if dose_multiplier is not None:
        # Simplistic model: larger insulin dose means lower glucose
        factor = 2 * (1 - dose_multiplier)  # If dose is doubled, reduce glucose by up to 100%
        modified_glucose = original_glucose * (1 + factor * 0.3)  # Scale the effect
    else:
        modified_glucose = original_glucose.copy()
    
    # Make a simple adjustment based on timing shift
    # Also very naive - just for demonstration
    if timing_shift is not None:
        # Later insulin (positive shift) means glucose goes higher before coming down
        # Earlier insulin (negative shift) means glucose stays lower
        if timing_shift > 0:
            # Add a bump in the beginning
            bump_size = timing_shift / 30.0  # Normalize to a fraction
            modified_glucose[:min(12, len(modified_glucose))] *= (1 + bump_size * 0.2)
        elif timing_shift < 0:
            # Lower glucose overall
            modified_glucose *= (1 - abs(timing_shift) / 60.0 * 0.1)
    
    return modified_glucose

# Create the evaluator
evaluator = CounterfactualEvaluator()

# Evaluate our dummy model
results = evaluator.evaluate_predictions(scenarios, dummy_model)

# Print the evaluation metrics
print("Evaluation results:")
print(f"Mean RMSE: {results['rmse'].mean():.2f} mg/dL")
print(f"Mean MAE: {results['mae'].mean():.2f} mg/dL")
print(f"Mean MAPE: {results['mape'].mean():.2f}%")

# Plot a few comparisons
print("\nGenerating visualizations for 3 example scenarios...")

# Select diverse scenarios to visualize
scenarios_to_plot = [
    # A timing shift scenario
    next(s for s in scenarios if s['timing_shift'] is not None and s['dose_multiplier'] is None),
    # A dose multiplier scenario
    next(s for s in scenarios if s['dose_multiplier'] is not None and s['timing_shift'] is None),
    # A combined scenario
    next(s for s in scenarios if s['dose_multiplier'] is not None and s['timing_shift'] is not None)
]

for i, scenario in enumerate(scenarios_to_plot):
    # Get the model's prediction
    prediction = dummy_model(
        scenario['input_data'], 
        scenario['intervention_time'],
        scenario['dose_multiplier'],
        scenario['timing_shift']
    )
    
    # Create comparison plot
    fig = evaluator.plot_comparison(scenario, prediction)
    
    # Add details to title
    intervention_type = []
    if scenario['dose_multiplier'] is not None:
        intervention_type.append(f"Dose: {scenario['dose_multiplier']:.2f}x")
    if scenario['timing_shift'] is not None:
        direction = "later" if scenario['timing_shift'] > 0 else "earlier"
        intervention_type.append(f"Timing: {abs(scenario['timing_shift'])}min {direction}")
    
    fig.update_layout(
        title=f"Example {i+1}: {' & '.join(intervention_type)}"
    )
    
    # Save the plot
    fig.write_html(f'example_visualization_{i+1}.html')

print("Done! Check the generated HTML files for visualizations.") 