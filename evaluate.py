import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the custom model
from src.model import TemporalFusionTransformer

def plot_forecast(y_true, y_pred_quantiles, time_steps):
    """Plots the actual values against the predicted P10, P50, and P90 funnels."""
    plt.figure(figsize=(12, 6))
    
    # Plot the ground truth
    plt.plot(time_steps, y_true, label='Actual Spend', color='black', linewidth=2)
    
    # Plot the Median (P50) prediction
    plt.plot(time_steps, y_pred_quantiles[:, 1], label='Median Forecast (P50)', color='blue')
    
    # Fill the area between P10 and P90 to show the confidence interval
    plt.fill_between(
        time_steps, 
        y_pred_quantiles[:, 0], # P10 (Lower bound)
        y_pred_quantiles[:, 2], # P90 (Upper bound)
        color='blue', alpha=0.2, label='80% Confidence Interval'
    )
    
    plt.title("Customer Lifetime Value Forecast")
    plt.xlabel("Days")
    plt.ylabel("Spend ($)")
    plt.legend()
    plt.savefig('forecast_plot.png')
    plt.show()

def plot_feature_importance(vsn_weights, feature_names):
    """Plots the Variable Selection Network weights as a bar chart."""
    # Average the weights across the batch and time steps
    mean_weights = np.mean(vsn_weights, axis=(0, 1, 3))
    
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, mean_weights, color='orange')
    plt.title("VSN Feature Importance (What the model cared about)")
    plt.ylabel("Attention Weight")
    plt.savefig('vsn_importance.png')
    plt.show()

def main():
    # 1. Re-instantiate the model structure
    past_features_cols = ['daily_spend', 'logins', 'clicks'] 
    future_features_cols = ['day_of_week', 'is_holiday']
    
    model = TemporalFusionTransformer(
        num_past_features=len(past_features_cols),
        num_future_features=len(future_features_cols),
        hidden_units=32,
        num_quantiles=3,
        num_heads=4
    )
    
    # 2. Build the model by passing a dummy batch of data through it
    dummy_past = tf.zeros((1, 30, len(past_features_cols)))
    dummy_future = tf.zeros((1, 7, len(future_features_cols)))
    model((dummy_past, dummy_future))
    
    # 3. Load the perfectly tuned weights we saved during train.py
    model.load_weights('saved_models/best_tft_weights.weights.h5')
    print("Model weights loaded successfully.")

    # --- FOR DEMONSTRATION: Assume we grabbed one test window ---
    # In reality, you'd pull this from your test tf.data.Dataset
    test_past = tf.random.normal((1, 30, 3))
    test_future = tf.random.normal((1, 7, 2))
    test_y_true = np.random.normal(50, 10, (7,)) # 7 days of actual future spend
    
    # 4. Generate Predictions
    predictions = model((test_past, test_future), training=False)
    
    # Extract the prediction for the 1st item in the batch
    # Shape becomes: (7 time_steps, 3 quantiles)
    y_pred_quantiles = predictions[0].numpy() 
    
    future_days = range(31, 38) # Days 31 to 37
    plot_forecast(test_y_true, y_pred_quantiles, future_days)
    
    # 5. Extract the "Brain State" (Interpretability)
    # We can manually pass the data through the past_vsn layer to get the weights
    past_list = tf.unstack(test_past, axis=-1)
    _, past_weights = model.past_vsn(past_list, training=False)
    
    plot_feature_importance(past_weights.numpy(), past_features_cols)
    
    print("Evaluation complete. Visualizations saved.")

if __name__ == "__main__":
    main()