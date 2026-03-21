import tensorflow as tf

class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, **kwargs):
        """
        Args:
            quantiles: A list of floats representing the quantiles to predict.
                       Example: [0.1, 0.5, 0.9] for 10th, 50th (median), and 90th percentiles.
        """
        super().__init__(**kwargs)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth values. Shape: (batch_size, time_steps, 1)
            y_pred: Predicted values for each quantile. Shape: (batch_size, time_steps, num_quantiles)
        """
        # Ensure y_true has the same number of dimensions as y_pred
        # Shape becomes: (batch_size, time_steps, 1)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        if len(y_true.shape) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)

        losses = []
        
        # Calculate the loss for each quantile independently
        for i, q in enumerate(self.quantiles):
            # Extract the prediction for this specific quantile
            # Shape: (batch_size, time_steps, 1)
            y_pred_q = tf.expand_dims(y_pred[..., i], axis=-1)
            
            # Calculate the error (residuals)
            error = y_true - y_pred_q
            
            # The core Quantile Loss (Pinball Loss) formula
            loss_q = tf.maximum(q * error, (q - 1.0) * error)
            
            losses.append(loss_q)
            
        # Stack all the losses: (batch_size, time_steps, num_quantiles)
        losses = tf.stack(losses, axis=-1)
        
        # Average the loss across all time steps and all quantiles
        return tf.reduce_mean(losses)