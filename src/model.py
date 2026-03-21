import tensorflow as tf
from .tft_layers import VariableSelectionNetwork, GatedResidualNetwork

class TemporalFusionTransformer(tf.keras.Model):
    def __init__(self, num_past_features, num_future_features, hidden_units, num_quantiles=3, num_heads=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.num_quantiles = num_quantiles
        
        # 1. Variable Selection Networks (Filters the noise)
        # One for the past observed data, one for the known future data
        self.past_vsn = VariableSelectionNetwork(num_past_features, hidden_units, dropout_rate)
        self.future_vsn = VariableSelectionNetwork(num_future_features, hidden_units, dropout_rate)
        
        # 2. Sequence Processing (Short-term sequential memory)
        # We use LSTM to process the filtered features step-by-step
        self.lstm_encoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
        self.lstm_decoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True)
        
        # 3. Multi-Head Attention (Long-term global memory)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)
        
        # 4. Final Processing & Output
        self.post_attention_grn = GatedResidualNetwork(hidden_units, dropout_rate)
        self.output_layer = tf.keras.layers.Dense(num_quantiles)

    def call(self, inputs, training=False):
        """
        inputs: A dictionary or tuple containing:
          - past_inputs: e.g., past sales, clicks (Batch, Past_Time_Steps, Past_Features)
          - future_inputs: e.g., upcoming holidays (Batch, Future_Time_Steps, Future_Features)
        """
        past_inputs, future_inputs = inputs
        
        # Step A: Filter out the noisy variables
        # Unpack the features into lists for the VSN
        # Step A: Filter out the noisy variables
        # Unpack the features into lists for the VSN using tf.split to preserve the 3D shape
        past_list = tf.split(past_inputs, past_inputs.shape[-1], axis=-1)
        future_list = tf.split(future_inputs, future_inputs.shape[-1], axis=-1)
        past_features, past_weights = self.past_vsn(past_list, training=training)
        future_features, future_weights = self.future_vsn(future_list, training=training)
        
        # Step B: Sequential processing (The LSTM block)
        # The encoder processes the past and saves its final state (like saving to a CPU register)
        encoded_past, state_h, state_c = self.lstm_encoder(past_features)
        
        # The decoder takes that saved state and uses it to start processing the future
        decoded_future = self.lstm_decoder(future_features, initial_state=[state_h, state_c])
        
        # Combine the sequential representations
        lstm_output = tf.concat([encoded_past, decoded_future], axis=1)
        
        # Step C: Multi-Head Attention (Looking back in time)
        # The model asks: "For each step in the future, which past steps are most relevant?"
        attention_output = self.attention(
            query=lstm_output, 
            value=lstm_output, 
            key=lstm_output, 
            training=training
        )
        
        # Step D: Final refinement and prediction
        # Add a residual connection around the attention layer
        x = lstm_output + attention_output
        x = self.post_attention_grn(x, training=training)
        
        # Output shape: (Batch, Total_Time_Steps, Num_Quantiles)
        # Output shape: (Batch, Total_Time_Steps, Num_Quantiles)
        quantiles = self.output_layer(x)
        
        # We only want to return the predictions for the future time steps!
        # We dynamically slice the last N steps off the tensor, where N is the length of our future inputs.
        future_len = tf.shape(future_inputs)[1]
        quantiles = quantiles[:, -future_len:, :]
        
        return quantiles