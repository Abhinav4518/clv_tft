import os
import pandas as pd
import numpy as np
from datetime import timedelta, date

# Set a random seed so your data is the same every time you run the script
np.random.seed(42)

def generate_synthetic_data(num_customers=50, days_per_customer=120):
    records = []
    # Starting the simulation from a fixed date
    start_date = date(2025, 1, 1)

    for cust_id in range(1, num_customers + 1):
        customer_name = f"C_{cust_id:03d}"
        
        # Assign a static tier to each customer with weighted probabilities
        tier = np.random.choice(['Gold', 'Silver', 'Bronze'], p=[0.2, 0.5, 0.3])
        spend_multiplier = {'Gold': 2.5, 'Silver': 1.2, 'Bronze': 0.8}[tier]

        for day in range(days_per_customer):
            current_date = start_date + timedelta(days=day)
            
            # Future Known Inputs
            day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
            is_holiday = 1 if np.random.random() < 0.04 else 0 # roughly 4% chance of a holiday
            
            # Past Observed Inputs
            # Logins follow a Poisson distribution (some days 0, some days 2-3)
            base_login_rate = 1.5 if tier == 'Gold' else 0.8
            logins = np.random.poisson(lam=base_login_rate)
            
            # Clicks are directly correlated to logins + some random browsing noise
            clicks = (logins * np.random.randint(2, 10)) + np.random.randint(0, 5)

            # Target Variable: Daily Spend
            if logins > 0:
                # Spend is correlated with clicks, multiplied by their tier status
                base_spend = clicks * np.random.uniform(0.5, 2.0) * spend_multiplier
                
                # Introduce weekly seasonality: 30% boost on weekends (Saturday/Sunday)
                if day_of_week >= 5:
                    base_spend *= 1.3
                    
                # Introduce external spikes: 50% boost on holidays
                if is_holiday:
                    base_spend *= 1.5
            else:
                # If they didn't log in, they didn't spend money
                base_spend = 0.0

            records.append({
                'customer_id': customer_name,
                'date': current_date.strftime("%Y-%m-%d"),
                'time_idx': day, # Continuous integer starting at 0 for each customer
                'daily_spend': round(base_spend, 2),
                'logins': logins,
                'clicks': clicks,
                'day_of_week': day_of_week,
                'is_holiday': is_holiday,
                'customer_tier': tier
            })

    return pd.DataFrame(records)

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("Generating synthetic customer data...")
    # Generate 120 days of history for 100 unique customers
    df = generate_synthetic_data(num_customers=100, days_per_customer=120)
    
    file_path = "data/raw_customer_data.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Success! Generated {len(df)} rows and saved to {file_path}")
    print("\nSample Data (First 5 Rows):")
    print(df.head())
