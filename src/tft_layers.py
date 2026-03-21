import tensorflow as tf

class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        # We multiply by 2 because we will split this tensor in half later
        self.dense = tf.keras.layers.Dense(units * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        # Split the output into two equal halves along the last axis
        activation_part, gate_part = tf.split(x, 2, axis=-1)
        
        # Apply the gating mechanism
        return activation_part * tf.nn.sigmoid(gate_part)


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        
        # The primary dense layer with Exponential Linear Unit activation
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='elu')
        
        # The secondary dense layer (linear)
        self.dense2 = tf.keras.layers.Dense(hidden_units)
        
        # The GLU to control information flow
        self.glu = GatedLinearUnit(hidden_units)
        
        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layer Normalization and an optional projection for the residual connection
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.project_residual = tf.keras.layers.Dense(hidden_units)

    def call(self, inputs, context=None, training=False):
        # Optional context vector (used later for static variables)
        if context is not None:
            context = tf.keras.layers.Dense(self.hidden_units)(context)
            # Add context to inputs before passing through the network
            x = self.dense1(inputs + context) 
        else:
            x = self.dense1(inputs)
            
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.glu(x)
        
        # Residual connection setup
        residual = self.project_residual(inputs) if inputs.shape[-1] != self.hidden_units else inputs
        
        # Add the residual and normalize
        return self.layer_norm(x + residual)
    
class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, num_features, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_units = hidden_units
        
        # 1. The network that decides the "importance" of each feature
        self.weight_network = GatedResidualNetwork(hidden_units, dropout_rate)
        self.weight_projection = tf.keras.layers.Dense(num_features)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        
        # 2. A dedicated processing network for EACH individual feature
        self.feature_networks = [
            GatedResidualNetwork(hidden_units, dropout_rate) 
            for _ in range(num_features)
        ]

    def call(self, inputs, context=None, training=False):
        # 'inputs' is expected to be a list of tensors, one for each feature column.
        # Each tensor has shape: (batch_size, time_steps, input_dim)
        
        # Step A: Figure out how much weight to give each feature
        concatenated_inputs = tf.concat(inputs, axis=-1)
        
        # The weight network looks at all features (and static context) together
        weight_outputs = self.weight_network(concatenated_inputs, context=context, training=training)
        weight_outputs = self.weight_projection(weight_outputs)
        
        # Softmax ensures all feature weights sum to 1.0 at every single time step
        weights = self.softmax(weight_outputs) 
        
        # Expand dimensions so we can multiply these weights against the feature matrices
        # Shape becomes: (batch_size, time_steps, num_features, 1)
        weights = tf.expand_dims(weights, axis=-1) 

        # Step B: Process each feature through its own private GRN
        processed_features = []
        for i in range(self.num_features):
            # Each feature gets complex, non-linear processing independent of the others
            processed = self.feature_networks[i](inputs[i], training=training)
            processed_features.append(processed)
            
        # Stack into a single tensor: (batch_size, time_steps, num_features, hidden_units)
        processed_features = tf.stack(processed_features, axis=2)

        # Step C: Apply the weights
        weighted_features = processed_features * weights
        
        # Step D: Collapse the features down into a single representation per time step
        # Shape becomes: (batch_size, time_steps, hidden_units)
        output = tf.reduce_sum(weighted_features, axis=2)
        
        # We return the weights alongside the output. This is crucial for interpretability!
        return output, weights