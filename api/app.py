from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime

# Import your custom architecture
from src import TemporalFusionTransformer, TFTDataFormatter

app = FastAPI(title="CLV Forecast API")

# Allow your Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your Vercel URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the model and data
model = None
df = None
formatter = None
past_features_cols = ['daily_spend', 'logins', 'clicks'] 
future_features_cols = ['day_of_week', 'is_holiday']

@app.on_event("startup")
async def load_brain():
    """Loads the model weights and CSV data into RAM when the server boots."""
    global model, df, formatter
    print("Booting up AI Engine...")
    
    # Initialize architecture
    model = TemporalFusionTransformer(
        num_past_features=len(past_features_cols),
        num_future_features=len(future_features_cols),
        hidden_units=32, num_quantiles=3, num_heads=4
    )
    
    # Run dummy data to build the graph, then load weights
    dummy_past = tf.zeros((1, 30, 3))
    dummy_future = tf.zeros((1, 7, 2))
    model((dummy_past, dummy_future))
    model.load_weights('saved_models/best_tft_weights.weights.h5')
    
    # Load data and fit the scaler
    df = pd.read_csv('data/raw_customer_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    formatter = TFTDataFormatter(30, 7, 'customer_id', 'daily_spend')
    formatter.scaler.fit(df[past_features_cols])
    
    print("Ready to serve predictions!")

@app.get("/predict/{customer_id}")
async def predict_clv(customer_id: str, target_date: str = None):
    """
    Takes a customer ID and an optional target date. 
    Dynamically fetches 30 days prior to the target date, and predicts the 7 days after.
    """
    cust_df = df[df['customer_id'] == customer_id].copy()
    if cust_df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    # 1. Determine the Anchor Date
    if target_date:
        anchor_date = pd.to_datetime(target_date)
    else:
        # Default to the 30th day from the very end of their history
        anchor_date = cust_df['date'].max() - pd.Timedelta(days=6)
        
    # 2. Slice the Data dynamically based on the Anchor Date
    past_data = cust_df[cust_df['date'] < anchor_date].tail(30)
    future_data = cust_df[cust_df['date'] >= anchor_date].head(7)
    
    # 3. Safety Checks
    if len(past_data) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough historical data before {target_date}. Need 30 days.")
    if len(future_data) < 7:
        raise HTTPException(status_code=400, detail=f"Not enough future data after {target_date}. Need 7 days.")
        
    # Re-combine temporarily so our scaler can process them together
    combined_data = pd.concat([past_data, future_data])
    combined_scaled = formatter.preprocess_and_scale(combined_data, past_features_cols)
    
    # 4. Create the final Tensors
    past_inputs = np.array([combined_scaled.head(30)[past_features_cols].values], dtype=np.float32)
    future_inputs = np.array([combined_scaled.tail(7)[future_features_cols].values], dtype=np.float32)
    
    # 5. Run the forecast
    predictions = model((past_inputs, future_inputs), training=False)
    y_pred_quantiles = predictions[0].numpy()
    
    # 6. Return as clean JSON with formatted dates
    return {
        "customer_id": customer_id,
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "forecast": [
            {
                "date": future_data.iloc[i]['date'].strftime("%b %d"),
                "p10_lower_bound": round(float(y_pred_quantiles[i][0]), 2),
                "p50_median": round(float(y_pred_quantiles[i][1]), 2),
                "p90_upper_bound": round(float(y_pred_quantiles[i][2]), 2)
            }
            for i in range(7)
        ]
    }