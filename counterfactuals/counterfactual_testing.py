import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from copy import deepcopy
import os
from pathlib import Path
import pickle
from simulation.simple_glucose_gen import EnhancedGlucoseGenerator
import torch
import matplotlib.dates as mdates
from matplotlib.patches import Patch

class CounterfactualScenarioGenerator:
    def __init__(self, seed=42):
        self.base_generator = EnhancedGlucoseGenerator(seed=seed)
        self.rng = np.random.default_rng(seed)
    
    def generate_base_data(self, days=10, start_date='2024-01-01'):
        """Generate baseline data for several days"""
        return self.base_generator.generate_data(days=days, start_date=start_date)
    
    def sample_intervention_windows(self, base_data, n_samples=20, 
                                    hours_before=4, hours_after=5, 
                                    require_meal=True):
        """
        Sample time windows for potential counterfactual interventions.
        
        Args:
            base_data: DataFrame with baseline glucose data
            n_samples: Number of samples to generate
            hours_before: Hours of data to include before intervention
            hours_after: Hours of data to include after intervention
            require_meal: If True, only sample windows with a meal+insulin event
            
        Returns:
            List of (start_time, intervention_time, end_time) tuples
        """
        samples = []
        
        # Find suitable intervention points (where insulin was given)
        if require_meal:
            # This will be datetime index values
            potential_points = base_data[base_data['insulin'] > 0].index.tolist()
        else:
            # Take regular samples throughout the day
            # Make sure we get datetime index values, not integer positions
            potential_points = base_data.index[::12*6].tolist()  # Every 6 hours (at 12 points per hour)
            
        # Shuffle to randomize
        potential_points = self.rng.permutation(potential_points)
        
        for point in potential_points:
            # Check if we have enough data before and after
            start_time = point - timedelta(hours=hours_before)
            end_time = point + timedelta(hours=hours_after)
            
            if start_time >= base_data.index[0] and end_time <= base_data.index[-1]:
                # Valid window
                samples.append((start_time, point, end_time))
                if len(samples) >= n_samples:
                    break
                    
        return samples
    
    def create_counterfactual_scenario(self, base_data, intervention_time, 
                                      dose_multiplier=None, timing_shift=None):
        """
        Create a counterfactual scenario by modifying the insulin dose or timing at the intervention point.
        
        Args:
            base_data: Original DataFrame with glucose data
            intervention_time: The time at which to modify insulin dose or timing
            dose_multiplier: Factor to multiply the insulin dose (e.g., 0.5, 1.5)
            timing_shift: Minutes to shift insulin timing (positive = later, negative = earlier)
            
        Returns:
            DataFrame with the counterfactual scenario data
        """
        # Create a deep copy of the base data
        cf_data = base_data.copy()
        
        # Find the exact intervention point (insulin dose)
        if intervention_time not in cf_data.index:
            intervention_time = cf_data.index[cf_data.index.get_indexer([intervention_time], method='nearest')[0]]
        
        if dose_multiplier is not None:
            # Modify the insulin dose
            original_dose = cf_data.loc[intervention_time, 'insulin']
            if original_dose > 0:
                cf_data.loc[intervention_time, 'insulin'] = original_dose * dose_multiplier
        
        if timing_shift is not None and timing_shift != 0:
            # Shift the insulin timing
            original_dose = cf_data.loc[intervention_time, 'insulin']
            if original_dose > 0:
                # Remove original dose
                cf_data.loc[intervention_time, 'insulin'] = 0
                
                # Determine new timing - convert numpy int to Python int
                timing_shift_int = int(timing_shift)
                new_time = intervention_time + timedelta(minutes=timing_shift_int)
                
                # Ensure the new time is in the index
                if new_time not in cf_data.index:
                    new_time = cf_data.index[cf_data.index.get_indexer([new_time], method='nearest')[0]]
                
                # Add dose at new time (add to existing if there's already insulin there)
                cf_data.loc[new_time, 'insulin'] += original_dose
        
        # Generate new glucose values using the modified inputs
        # The key is to recreate the glucose curve with the counterfactual insulin
        return self._regenerate_glucose_curve(cf_data)
    
    def _regenerate_glucose_curve(self, modified_df):
        """
        Regenerate glucose values based on the modified insulin/carbs inputs.
        This replicates the glucose calculation logic from EnhancedGlucoseGenerator.
        """
        # Create new dataframe with recalculated glucose
        result_df = modified_df.copy()
        
        # Reset glucose to initial value
        result_df['glucose'] = self.base_generator.params['basal_glucose']
        
        # Pre-calculate insulin and carb effects
        insulin_activity = np.zeros(len(result_df))
        carb_impact = np.zeros(len(result_df))
        
        # Calculate glucose dynamics
        for t in range(1, len(result_df)):
            current_time = result_df.index[t]
            minutes_since_midnight = (current_time.hour * 60 + current_time.minute)
            
            # Calculate lagged insulin effects
            for past_t in range(max(0, t - self.base_generator.params['insulin_duration']//5), t):
                if result_df['insulin'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5  # Convert steps to minutes
                    insulin_activity[t] += self.base_generator._insulin_curve(time_diff, result_df['insulin'].iloc[past_t])
            
            # Calculate lagged carb effects
            for past_t in range(max(0, t - self.base_generator.params['carb_duration']//5), t):
                if result_df['carbs'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5  # Convert steps to minutes
                    carb_impact[t] += self.base_generator._carb_curve(time_diff, result_df['carbs'].iloc[past_t])
            
            # Calculate current glucose with all effects
            exercise_effect = 1 - (result_df['exercise'].iloc[t] * self.base_generator.params['exercise_sensitivity'] / 100)
            stress_effect = result_df['stress'].iloc[t] * self.base_generator.params['stress_effect']
            dawn_effect = self.base_generator._dawn_effect(current_time.hour + current_time.minute/60)
            
            # Combine all effects with appropriate scaling and momentum
            target_glucose = (
                self.base_generator.params['basal_glucose']
                + carb_impact[t] * self.base_generator.params['carb_impact']
                - insulin_activity[t] * self.base_generator.params['insulin_sensitivity'] * exercise_effect
                + stress_effect
                + dawn_effect
                + self.rng.normal(0, self.base_generator.params['noise_level'])
            )
            
            # Add momentum (glucose doesn't change instantly)
            result_df.iloc[t, result_df.columns.get_loc('glucose')] = 0.9 * result_df.iloc[t-1, result_df.columns.get_loc('glucose')] + 0.1 * target_glucose
        
        # Store calculated activity values
        result_df['active_insulin'] = insulin_activity
        result_df['carb_impact'] = carb_impact
        
        # Ensure glucose stays within realistic bounds
        result_df['glucose'] = np.clip(result_df['glucose'], 40, 400)
        
        return result_df

    def generate_test_scenarios(self, n_samples=20, hours_before=4, hours_after=5):
        """
        Generate a set of test scenarios with original and counterfactual data.
        
        Returns:
            List of dictionaries with test scenarios
        """
        # Generate base data
        base_data = self.generate_base_data(days=30)
        
        # Sample intervention windows
        windows = self.sample_intervention_windows(
            base_data, 
            n_samples=n_samples, 
            hours_before=hours_before,
            hours_after=hours_after
        )
        
        test_scenarios = []
        
        for i, (start_time, intervention_time, end_time) in enumerate(windows):
            # Extract the window data
            window_data = base_data.loc[start_time:end_time].copy()
            
            # Create a counterfactual scenario
            # Randomly choose modification type
            mod_type = self.rng.choice(['dose', 'timing', 'both'])
            
            dose_multiplier = None
            timing_shift = None
            
            if mod_type in ['dose', 'both']:
                # Modify dose by 0.5x to 2x
                dose_multiplier = float(self.rng.uniform(0.5, 2.0))
                
            if mod_type in ['timing', 'both']:
                # Shift timing by -30 to +30 minutes, in 5-minute increments
                # Convert to standard Python int to avoid numpy.int64 issues
                timing_shift = int(self.rng.choice(list(range(-30, 35, 5))))
            
            # Generate counterfactual
            cf_data = self.create_counterfactual_scenario(
                window_data, 
                intervention_time,
                dose_multiplier=dose_multiplier,
                timing_shift=timing_shift
            )
            
            # Split into input (before intervention) and output (after intervention)
            input_data = window_data.loc[:intervention_time].copy()
            true_output = window_data.loc[intervention_time:end_time].copy()
            cf_output = cf_data.loc[intervention_time:end_time].copy()
            
            # Create a test scenario
            scenario = {
                'id': i,
                'start_time': start_time,
                'intervention_time': intervention_time,
                'end_time': end_time,
                'input_data': input_data,
                'original_output': true_output,
                'counterfactual_output': cf_output,
                'dose_multiplier': dose_multiplier,
                'timing_shift': timing_shift,
                'original_dose': window_data.loc[intervention_time, 'insulin'] if intervention_time in window_data.index else 0,
                'counterfactual_dose': cf_data.loc[intervention_time, 'insulin'] if intervention_time in cf_data.index else 0
            }
            
            test_scenarios.append(scenario)
            
        return test_scenarios

class CounterfactualEvaluator:
    """Evaluates counterfactual prediction accuracy"""
    
    def __init__(self):
        pass
    
    def evaluate_predictions(self, test_scenarios, prediction_fn):
        """
        Evaluate predictions against ground truth counterfactual data.
        
        Args:
            test_scenarios: List of test scenario dictionaries
            prediction_fn: Function that takes (input_data, intervention_time, 
                          dose_multiplier, timing_shift) and returns predicted glucose values
                          
        Returns:
            DataFrame with evaluation metrics for each scenario
        """
        results = []
        
        for scenario in test_scenarios:
            # Get the model's prediction for this scenario
            predicted_glucose = prediction_fn(
                scenario['input_data'], 
                scenario['intervention_time'],
                scenario['dose_multiplier'],
                scenario['timing_shift']
            )
            
            # The ground truth is in scenario['counterfactual_output']['glucose']
            ground_truth = scenario['counterfactual_output']['glucose'].values
            
            # Ensure predictions and ground truth have the same length
            if len(predicted_glucose) != len(ground_truth):
                # Trim or pad as needed
                min_len = min(len(predicted_glucose), len(ground_truth))
                predicted_glucose = predicted_glucose[:min_len]
                ground_truth = ground_truth[:min_len]
            
            # Calculate evaluation metrics
            rmse = np.sqrt(np.mean((predicted_glucose - ground_truth)**2))
            mae = np.mean(np.abs(predicted_glucose - ground_truth))
            mape = np.mean(np.abs((predicted_glucose - ground_truth) / ground_truth)) * 100
            
            # Store results
            results.append({
                'scenario_id': scenario['id'],
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'dose_multiplier': scenario['dose_multiplier'],
                'timing_shift': scenario['timing_shift']
            })
            
        return pd.DataFrame(results)
    
    def plot_comparison(self, scenario, prediction=None):
        """
        Plot original vs counterfactual glucose curves for a given scenario.
        
        Args:
            scenario: Test scenario dictionary
            prediction: Optional predicted glucose values
        """
        fig = go.Figure()
        
        # Combine all data for plotting
        plot_df = pd.DataFrame(index=pd.date_range(
            scenario['start_time'], 
            scenario['end_time'], 
            freq='5min'
        ))
        
        # Original data
        plot_df['original'] = np.nan
        orig_data = pd.concat([scenario['input_data'], scenario['original_output']])
        plot_df.loc[orig_data.index, 'original'] = orig_data['glucose']
        
        # Counterfactual data
        plot_df['counterfactual'] = np.nan
        cf_output = scenario['counterfactual_output']
        plot_df.loc[cf_output.index, 'counterfactual'] = cf_output['glucose']
        
        # Mark pre-intervention and post-intervention
        plot_df['phase'] = 'pre'
        plot_df.loc[scenario['intervention_time']:, 'phase'] = 'post'
        
        # Add model prediction if provided
        if prediction is not None:
            plot_df['prediction'] = np.nan
            
            # If prediction is a numpy array, we need to align it with the index
            if isinstance(prediction, np.ndarray):
                # Get the post-intervention indices
                post_indices = plot_df.index[plot_df['phase'] == 'post']
                
                # Ensure the prediction length matches the number of post-intervention points
                if len(prediction) >= len(post_indices):
                    # Use only as many predictions as we have points
                    for i, idx in enumerate(post_indices):
                        if i < len(prediction):
                            plot_df.loc[idx, 'prediction'] = prediction[i]
                else:
                    # Pad with NaN if we have fewer predictions than points
                    for i, idx in enumerate(post_indices[:len(prediction)]):
                        plot_df.loc[idx, 'prediction'] = prediction[i]
            else:
                # If it's already aligned with the index (e.g., a Series), use it directly
                plot_df.loc[scenario['intervention_time']:, 'prediction'] = prediction
        
        # Plot original glucose curve
        fig.add_trace(go.Scatter(
            x=plot_df.index, 
            y=plot_df['original'],
            name='Original',
            line=dict(color='blue', width=2)
        ))
        
        # Plot counterfactual glucose curve
        fig.add_trace(go.Scatter(
            x=plot_df.index, 
            y=plot_df['counterfactual'],
            name='Counterfactual Ground Truth',
            line=dict(color='green', width=2)
        ))
        
        # Plot prediction if provided
        if prediction is not None:
            fig.add_trace(go.Scatter(
                x=plot_df.index, 
                y=plot_df['prediction'],
                name='Predicted Counterfactual',
                line=dict(color='red', dash='dash', width=2)
            ))
        
        # Mark intervention point
        int_glucose = plot_df.loc[scenario['intervention_time'], 'original']
        fig.add_trace(go.Scatter(
            x=[scenario['intervention_time']],
            y=[int_glucose],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name='Intervention Point'
        ))
        
        # Add insulin dose markers
        if 'insulin' in scenario['input_data'].columns:
            # Original insulin
            insulin_times = scenario['input_data'][scenario['input_data']['insulin'] > 0].index
            insulin_doses = scenario['input_data'].loc[insulin_times, 'insulin']
            insulin_glucose = scenario['input_data'].loc[insulin_times, 'glucose']
            
            fig.add_trace(go.Scatter(
                x=insulin_times,
                y=insulin_glucose,
                mode='markers',
                name='Original Insulin',
                marker=dict(
                    color='purple',
                    symbol='triangle-down',
                    size=insulin_doses*2 + 8
                ),
                text=[f'{d:.1f}u insulin' for d in insulin_doses],
                hovertemplate='%{text}<br>Glucose: %{y:.0f} mg/dL'
            ))
            
            # Counterfactual insulin after intervention
            if scenario['timing_shift'] is not None or scenario['dose_multiplier'] is not None:
                # Find intervention insulin
                if scenario['intervention_time'] in insulin_times:
                    # Find the modified insulin dose
                    if scenario['timing_shift'] is not None:
                        # Find shifted time
                        new_time = scenario['intervention_time'] + timedelta(minutes=scenario['timing_shift'])
                        cf_insulin_times = [new_time]
                        cf_insulin_doses = [scenario['original_dose']]
                        cf_glucose = [plot_df.loc[new_time, 'counterfactual'] if new_time in plot_df.index else None]
                    else:
                        cf_insulin_times = [scenario['intervention_time']]
                        cf_insulin_doses = [scenario['counterfactual_dose']]
                        cf_glucose = [plot_df.loc[scenario['intervention_time'], 'counterfactual']]
                    
                    fig.add_trace(go.Scatter(
                        x=cf_insulin_times,
                        y=cf_glucose,
                        mode='markers',
                        name='Counterfactual Insulin',
                        marker=dict(
                            color='orange',
                            symbol='triangle-down',
                            size=[d*2 + 8 for d in cf_insulin_doses]
                        ),
                        text=[f'{d:.1f}u insulin' for d in cf_insulin_doses],
                        hovertemplate='%{text}<br>Glucose: %{y:.0f} mg/dL'
                    ))
        
        # Add range guidelines
        fig.add_hline(y=180, line=dict(color='red', dash='dash', width=1))
        fig.add_hline(y=70, line=dict(color='red', dash='dash', width=1))
        fig.add_hline(y=100, line=dict(color='green', dash='dot', width=1))
        
        # Add vertical line at intervention time
        fig.add_vline(x=scenario['intervention_time'], line=dict(color='grey', dash='dot'))
        
        # Add scenario details
        title = "Counterfactual Scenario"
        if scenario['dose_multiplier'] is not None:
            title += f" - Dose {scenario['dose_multiplier']:.2f}x"
        if scenario['timing_shift'] is not None:
            direction = "later" if scenario['timing_shift'] > 0 else "earlier"
            title += f" - Timing {abs(scenario['timing_shift'])}min {direction}"
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Glucose (mg/dL)',
            hovermode='x unified',
            showlegend=True,
            yaxis=dict(range=[40, 300]),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

def save_test_scenarios(scenarios, output_dir='counterfactual_test_data'):
    """Save test scenarios to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scenarios
    with open(f"{output_dir}/test_scenarios.pkl", 'wb') as f:
        pickle.dump(scenarios, f)
    
    # Create a summary CSV
    summary = []
    for s in scenarios:
        summary.append({
            'id': s['id'],
            'start_time': s['start_time'],
            'intervention_time': s['intervention_time'],
            'end_time': s['end_time'],
            'dose_multiplier': s['dose_multiplier'],
            'timing_shift': s['timing_shift'],
            'original_dose': s['original_dose'],
            'counterfactual_dose': s['counterfactual_dose']
        })
    
    pd.DataFrame(summary).to_csv(f"{output_dir}/scenarios_summary.csv", index=False)
    
    print(f"Saved {len(scenarios)} test scenarios to {output_dir}")

def load_test_scenarios(input_dir='counterfactual_test_data'):
    """Load test scenarios from disk"""
    with open(f"{input_dir}/test_scenarios.pkl", 'rb') as f:
        scenarios = pickle.load(f)
    
    print(f"Loaded {len(scenarios)} test scenarios from {input_dir}")
    return scenarios

def dummy_prediction_function(input_data, intervention_time, dose_multiplier, timing_shift):
    """
    A dummy prediction function that just returns the original glucose values.
    This should be replaced with the actual ML model.
    """
    # Get the original glucose curve after intervention
    return input_data.loc[intervention_time:, 'glucose'].values

def convert_glucose_data_to_t4_format(data_window, normalize_insulin=True, max_insulin_dose=20):
    """
    Convert glucose data from the synthetic generator format to a format compatible with the T4 model.
    
    Args:
        data_window: DataFrame with glucose data for a specific window
        normalize_insulin: Whether to normalize insulin values to [0,1] range
        max_insulin_dose: The maximum insulin dose to use for normalization
        
    Returns:
        Dictionary with T4-compatible data including:
        - features: List of feature names
        - x: Feature values, shape [seq_len, n_features]
        - treatment: Treatment values for insulin, shape [seq_len, 2] where:
            - [:, 0] is normalized insulin dose (0-1)
            - [:, 1] is timing information (0=early, 1=late) derived from meal_insulin_delay
        - outcome: Blood glucose values
    """
    # Create copies to avoid modifying original data
    df = data_window.copy()
    
    # Extract the relevant columns for the T4 model
    # Core features: glucose, carbs, exercise, stress, active_insulin, carb_impact
    features = ['glucose', 'carbs', 'exercise', 'stress', 'active_insulin', 'carb_impact']
    x = df[features].values
    
    # Get outcome (glucose levels for next steps)
    outcome = df['glucose'].values
    
    # Process treatments (insulin and timing)
    treatments = np.zeros((len(df), 2))
    
    # First dimension: normalized insulin doses
    if normalize_insulin:
        treatments[:, 0] = np.clip(df['insulin'].values / max_insulin_dose, 0, 1)
    else:
        treatments[:, 0] = df['insulin'].values
    
    # Second dimension: timing information (based on meal_insulin_delay)
    # Convert meal_insulin_delay to a normalized timing value between 0 and 1
    # Negative values (insulin before meal) -> <0.5
    # Positive values (insulin after meal) -> >0.5
    # 0 (insulin at meal time) -> 0.5
    
    # Get the meal_insulin_delay values
    delay_values = df['meal_insulin_delay'].values
    
    # Map delays to [0,1] range: -30 min (early) → 0, 0 min (exact) → 0.5, +30 min (late) → 1
    normalized_delays = np.clip(delay_values / 60 + 0.5, 0, 1)  # Assuming ±30 minutes range
    
    # Fill in timing values where insulin was given
    for i in range(len(df)):
        if df['insulin'].values[i] > 0:
            treatments[i, 1] = normalized_delays[i] 
            
    # For positions where no insulin given, keep timing at 0
    
    # Create the data dictionary in T4 format
    t4_data = {
        'features': features,
        'x': x,
        'treatment': treatments,
        'outcome': outcome,
        # Add demographic data (dummy values, should be replaced with actual data if available)
        'agegroup': 0.5,  # normalized age
        'heightgroup': 0.5,  # normalized height
        'weightgroup': 0.5,  # normalized weight
        'gender': 0.0,  # binary gender 
        'death': 0.0  # binary outcome, not relevant for glucose data
    }
    
    return t4_data

def generate_glucose_counterfactuals(model, data_window, intervention_time, 
                              dose_multiplier=None, timing_shift=None,
                              device='cuda', pre_window=5, max_stay=24):
    """
    Generate counterfactual predictions for blood glucose levels using the T4 model.
    
    Args:
        model: Trained T4 model
        data_window: DataFrame with glucose data
        intervention_time: Time point at which to modify insulin
        dose_multiplier: Factor to multiply the insulin dose (e.g., 0.5, 1.5, None=no change)
        timing_shift: Minutes to shift insulin timing (e.g., -15, +15, None=no change)
        device: Device to run model on ('cuda' or 'cpu')
        pre_window: Number of future timesteps to predict
        max_stay: Maximum sequence length for model
        
    Returns:
        Dictionary with factual and counterfactual predictions
    """
    # Convert data to T4 format
    t4_data = convert_glucose_data_to_t4_format(data_window)
    
    # Get the index of the intervention time
    intervention_idx = data_window.index.get_indexer([intervention_time], method='nearest')[0]
    
    # Create factual input sequence
    x = t4_data['x'][:intervention_idx+1]
    x_demo = np.array([t4_data['agegroup'], t4_data['heightgroup'], 
                     t4_data['weightgroup'], t4_data['gender']])
    treatment = t4_data['treatment'][:intervention_idx+1]
    
    # Get original insulin value at intervention point
    original_dose = treatment[intervention_idx, 0]
    original_timing = treatment[intervention_idx, 1]
    
    # Create counterfactual treatment sequence
    cf_treatment = treatment.copy()
    
    # Apply dose modification if specified
    if dose_multiplier is not None and original_dose > 0:
        cf_treatment[intervention_idx, 0] = min(original_dose * dose_multiplier, 1.0)
    
    # Apply timing modification if specified
    if timing_shift is not None and original_dose > 0:
        # Calculate new timing value: shift by timing_shift minutes
        # Since timing is normalized to [0,1] where 0.5 = exact timing,
        # we need to convert minutes to this normalized scale
        timing_shift_normalized = timing_shift / 60  # Convert to hours (±0.5 = ±30 min)
        new_timing = original_timing + timing_shift_normalized
        cf_treatment[intervention_idx, 1] = np.clip(new_timing, 0, 1)
    
    # Ensure input dimensions match model expectations
    if len(x) < max_stay:
        # Pad with zeros if needed
        pad_size = max_stay - len(x)
        x_pad = np.pad(x, ((0, pad_size), (0, 0)), 'constant')
        treatment_pad = np.pad(treatment, ((0, pad_size), (0, 0)), 'constant')
        cf_treatment_pad = np.pad(cf_treatment, ((0, pad_size), (0, 0)), 'constant')
        mask = np.zeros((max_stay, x.shape[1]))
        mask[:len(x)] = 1
    else:
        # Truncate if too long
        x_pad = x[-max_stay:]
        treatment_pad = treatment[-max_stay:]
        cf_treatment_pad = cf_treatment[-max_stay:]
        mask = np.ones((max_stay, x.shape[1]))
    
    # Create dummy outputs for model input
    dummy_y = np.zeros((pre_window + 1))
    
    # Convert to tensors
    x_tensor = torch.tensor(x_pad, dtype=torch.float32).unsqueeze(0).to(device)
    x_demo_tensor = torch.tensor(x_demo, dtype=torch.float32).unsqueeze(0).to(device)
    treatment_tensor = torch.tensor(treatment_pad, dtype=torch.float32).unsqueeze(0).to(device)
    cf_treatment_tensor = torch.tensor(cf_treatment_pad, dtype=torch.float32).unsqueeze(0).to(device)
    y_tensor = torch.tensor(dummy_y, dtype=torch.float32).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run model for factual prediction
    with torch.no_grad():
        factual_output, _, _, _, _ = model(x_tensor, y_tensor, x_demo_tensor, treatment_tensor)
    
    # Run model for counterfactual prediction
    with torch.no_grad():
        _, counterfactual_output, _, _, _ = model(x_tensor, y_tensor, x_demo_tensor, cf_treatment_tensor)
    
    # Convert predictions back to numpy and reshape
    factual_pred = factual_output.cpu().numpy()[0]
    counterfactual_pred = counterfactual_output.cpu().numpy()[0]
    
    # The full original and counterfactual trajectories
    factual_trajectory = np.concatenate([x[:, 0], factual_pred])
    counterfactual_trajectory = np.concatenate([x[:, 0], counterfactual_pred])
    
    # Create timestamps for the predictions
    pred_timestamps = pd.date_range(
        start=data_window.index[intervention_idx], 
        periods=pre_window+1, 
        freq=pd.infer_freq(data_window.index)
    )[1:]
    
    # Create DataFrames with the predictions
    factual_df = pd.DataFrame(
        index=pred_timestamps,
        data={'glucose': factual_pred}
    )
    
    counterfactual_df = pd.DataFrame(
        index=pred_timestamps,
        data={'glucose': counterfactual_pred}
    )
    
    return {
        'intervention_time': intervention_time,
        'original_dose': original_dose,
        'original_timing': original_timing,
        'counterfactual_dose': cf_treatment[intervention_idx, 0],
        'counterfactual_timing': cf_treatment[intervention_idx, 1],
        'factual_predictions': factual_df,
        'counterfactual_predictions': counterfactual_df,
        'factual_trajectory': factual_trajectory,
        'counterfactual_trajectory': counterfactual_trajectory
    }

def plot_glucose_counterfactuals(data_window, counterfactual_results, 
                              highlight_range=(70, 180), figsize=(12, 8)):
    """
    Create a detailed plot of glucose counterfactual analysis.
    
    Args:
        data_window: Original DataFrame with glucose data
        counterfactual_results: Dictionary with counterfactual predictions from generate_glucose_counterfactuals
        highlight_range: Tuple of (min, max) values for the target glucose range
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure object
    """
    intervention_time = counterfactual_results['intervention_time']
    
    # Extract data for the plot
    original_data = data_window.loc[:intervention_time].copy()
    factual_pred = counterfactual_results['factual_predictions'].copy()
    cf_pred = counterfactual_results['counterfactual_predictions'].copy()
    
    # Combine data for plotting
    combined_data = pd.concat([
        original_data[['glucose', 'insulin', 'carbs']],
        pd.DataFrame(index=factual_pred.index, 
                     data={'glucose_factual': factual_pred['glucose'].values})
    ])
    
    combined_data['glucose_cf'] = np.nan
    combined_data.loc[cf_pred.index, 'glucose_cf'] = cf_pred['glucose'].values
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot historical and predicted glucose values
    ax1.plot(combined_data.index, combined_data['glucose'], 'b-', label='Historical Glucose', linewidth=2)
    ax1.plot(combined_data.index, combined_data['glucose_factual'], 'b--', label='Factual Prediction', linewidth=2)
    ax1.plot(combined_data.index, combined_data['glucose_cf'], 'r--', label='Counterfactual Prediction', linewidth=2)
    
    # Mark intervention point
    ax1.axvline(x=intervention_time, color='purple', linestyle='--', linewidth=1.5)
    ax1.annotate('Intervention', 
                xy=(mdates.date2num(intervention_time), combined_data.loc[intervention_time, 'glucose']),
                xytext=(15, 15),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=10)
    
    # Add target range
    ax1.axhspan(highlight_range[0], highlight_range[1], color='green', alpha=0.1, label='Target Range')
    
    # Insulin and carbs in second subplot
    bar_width = 0.0025  # Width in date units
    insulin_data = combined_data['insulin']
    insulin_times = combined_data.index[insulin_data > 0]
    insulin_values = insulin_data[insulin_data > 0]
    carb_data = combined_data['carbs']
    carb_times = combined_data.index[carb_data > 0]
    carb_values = carb_data[carb_data > 0]
    
    # Plot original insulin doses
    for t, v in zip(insulin_times, insulin_values):
        ax2.bar(t, v, width=bar_width, color='red', alpha=0.7)
    
    # Plot original carb intake
    for t, v in zip(carb_times, carb_values):
        ax2.bar(t, v, width=bar_width, color='green', alpha=0.7)
    
    # Mark the counterfactual insulin
    cf_dose = counterfactual_results['counterfactual_dose']
    original_dose = counterfactual_results['original_dose']
    
    if cf_dose != original_dose:
        ax2.bar(intervention_time, cf_dose * 20, width=bar_width, color='magenta', alpha=0.9, label='CF Insulin')
    
    # Mark the timing difference if applicable
    cf_timing = counterfactual_results['counterfactual_timing']
    original_timing = counterfactual_results['original_timing']
    
    if cf_timing != original_timing:
        # Convert normalized timing (0-1) to minutes relative to meal (-30 to +30)
        timing_diff = (cf_timing - 0.5) * 60  # in minutes
        timing_text = f"{timing_diff:+.0f} min"
        
        ax2.annotate(timing_text,
                    xy=(mdates.date2num(intervention_time), cf_dose * 20),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)
    
    # Add formatting
    ax1.set_title('Glucose Counterfactual Analysis', fontsize=14)
    ax1.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Insulin (Units) / Carbs (g)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add a legend for insulin/carbs
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Insulin'),
        Patch(facecolor='green', alpha=0.7, label='Carbs')
    ]
    
    if cf_dose != original_dose:
        legend_elements.append(Patch(facecolor='magenta', alpha=0.9, label='CF Insulin'))
        
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Format x-axis to show time properly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Add summary information as text
    effect_size = np.mean(cf_pred['glucose'].values - factual_pred['glucose'].values)
    max_effect = np.max(np.abs(cf_pred['glucose'].values - factual_pred['glucose'].values))
    
    info_text = (
        f"Intervention: {'↑' if cf_dose > original_dose else '↓'} Insulin dose "
        f"({original_dose*20:.1f}u → {cf_dose*20:.1f}u)\n"
        f"Timing change: {(cf_timing - original_timing) * 60:+.0f} min\n"
        f"Mean effect: {effect_size:.1f} mg/dL\n"
        f"Max effect: {max_effect:.1f} mg/dL"
    )
    
    # Add annotation box
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Create generator
    generator = CounterfactualScenarioGenerator(seed=42)
    
    # Generate test scenarios
    print("Generating test scenarios...")
    test_scenarios = generator.generate_test_scenarios(n_samples=30)
    
    # Save the scenarios
    save_test_scenarios(test_scenarios)
    
    # Create visualizations directory
    os.makedirs('counterfactual_visualizations', exist_ok=True)
    
    # Plot a few examples
    print("Creating example visualizations...")
    evaluator = CounterfactualEvaluator()
    
    for i in range(min(5, len(test_scenarios))):
        scenario = test_scenarios[i]
        fig = evaluator.plot_comparison(scenario)
        fig.write_html(f'counterfactual_visualizations/scenario_{i}.html')
    
    print("Done! You can now use these test scenarios to evaluate your counterfactual prediction model.")
    # print("Example usage:")
    # print("1. Load the test scenarios: scenarios = load_test_scenarios()")
    # print("2. Evaluate your model: evaluator = CounterfactualEvaluator(); results = evaluator.evaluate_predictions(scenarios, your_model_prediction_function)") 