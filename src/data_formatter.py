import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class TFTDataFormatter:
    def __init__(self, encoder_steps, decoder_steps, id_column, target_column):
        """
        Args:
            encoder_steps: How many past time steps to look at (e.g., 30 days).
            decoder_steps: How many future time steps to predict (e.g., 7 days).
            id_column: The column identifying unique entities (e.g., 'customer_id').
            target_column: What we are trying to predict (e.g., 'daily_spend').
        """
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.total_window_size = encoder_steps + decoder_steps
        
        self.id_column = id_column
        self.target_column = target_column
        self.scaler = StandardScaler()

    def preprocess_and_scale(self, df, continuous_cols):
        """Scales continuous features to have zero mean and unit variance."""
        df_scaled = df.copy()
        df_scaled[continuous_cols] = self.scaler.fit_transform(df[continuous_cols])
        return df_scaled

    def _extract_windows(self, group_df, feature_cols):
        """Extracts sliding windows for a single customer."""
        data = group_df[feature_cols].values
        targets = group_df[self.target_column].values
        
        n_samples = len(data) - self.total_window_size + 1
        if n_samples <= 0:
            return [], []

        X, y = [], []
        # Slide a window across this customer's timeline
        for i in range(n_samples):
            # The input features for the entire window (past + future knowns)
            window_x = data[i : i + self.total_window_size]
            
            # The target values we want to predict (only the future part)
            # We slice from encoder_steps to the end of the window
            window_y = targets[i + self.encoder_steps : i + self.total_window_size]
            
            X.append(window_x)
            y.append(window_y)
            
        return X, y

    def build_tf_dataset(self, df, feature_cols, batch_size=32, shuffle=True):
        """Builds the final tf.data.Dataset optimized for training."""
        all_X, all_y = [], []
        
        # We MUST group by customer ID. We cannot let a window overlap 
        # the end of Customer A's timeline and the start of Customer B's.
        for _, group in df.groupby(self.id_column):
            # Sort chronologically just to be safe
            group = group.sort_index() 
            X, y = self._extract_windows(group, feature_cols)
            all_X.extend(X)
            all_y.extend(y)
            
        # Convert lists to 3D numpy arrays
        X_array = np.array(all_X, dtype=np.float32)
        y_array = np.array(all_y, dtype=np.float32)
        
        # Create the highly efficient tf.data pipeline
        dataset = tf.data.Dataset.from_tensor_slices((X_array, y_array))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X_array))
            
        # Batch and prefetch for GPU optimization
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset