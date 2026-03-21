import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Import our custom modules
from src.data_formatter import TFTDataFormatter
from src.model import TemporalFusionTransformer
from src.quantile_loss import QuantileLoss

def main():
    # 1. Define Hyperparameters
    ENCODER_STEPS = 30
    DECODER_STEPS = 7
    BATCH_SIZE = 64
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10 # 10 epochs is enough to prove the architecture works
    QUANTILES = [0.1, 0.5, 0.9]

    print("Loading and formatting data...")
    # 2. Load the synthetic data we generated
    df = pd.read_csv('data/raw_customer_data.csv')
    
    past_features_cols = ['daily_spend', 'logins', 'clicks'] 
    future_features_cols = ['day_of_week', 'is_holiday']
    target_col = 'daily_spend'
    id_col = 'customer_id'

    # Simple Train/Validation Split (Keep the last few days of each customer for testing)
    # Simple Train/Validation Split
    # We must ensure val_df has at least 37 days to form full sliding windows!
    train_df = df[df['time_idx'] < 80].copy()  # First 80 days for training
    val_df = df[df['time_idx'] >= 80].copy()   # Last 40 days for validation

    # 3. Initialize Formatter and Scale Data
    formatter = TFTDataFormatter(
        encoder_steps=ENCODER_STEPS, 
        decoder_steps=DECODER_STEPS, 
        id_column=id_col, 
        target_column=target_col
    )
    
    # Fit scaler only on training data to prevent data leakage
    formatter.scaler.fit(train_df[past_features_cols])
    train_df = formatter.preprocess_and_scale(train_df, past_features_cols)
    val_df = formatter.preprocess_and_scale(val_df, past_features_cols)

    # 4. Build Base Datasets
    all_features = past_features_cols + future_features_cols
    train_ds = formatter.build_tf_dataset(train_df, all_features, BATCH_SIZE)
    val_ds = formatter.build_tf_dataset(val_df, all_features, BATCH_SIZE, shuffle=False)

    # 5. Map the dataset to fit our custom model's exact input requirements
    def split_inputs(x, y):
        # x is shape (Batch, 37, 5)
        # We split it into past (Batch, 30, 3) and future (Batch, 7, 2)
        past_inputs = x[:, :ENCODER_STEPS, :len(past_features_cols)]
        future_inputs = x[:, ENCODER_STEPS:, len(past_features_cols):]
        return (past_inputs, future_inputs), y

    # tf.data.AUTOTUNE lets TensorFlow multithread this slicing process on your CPU
    train_ds = train_ds.map(split_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(split_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    print("Initializing Temporal Fusion Transformer...")
    # 6. Instantiate and Compile Model
    model = TemporalFusionTransformer(
        num_past_features=len(past_features_cols),
        num_future_features=len(future_features_cols),
        hidden_units=HIDDEN_UNITS,
        num_quantiles=len(QUANTILES),
        num_heads=4,
        dropout_rate=0.1
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = QuantileLoss(quantiles=QUANTILES)
    model.compile(optimizer=optimizer, loss=loss_fn)

    os.makedirs('saved_models', exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_models/best_tft_weights.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]

    print("Starting actual training loop...")
    # 7. Execute Training for real!
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("Training complete. Best weights successfully saved to disk.")

if __name__ == "__main__":
    main()