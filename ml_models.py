import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from datetime import datetime

class TechSightML:
    def __init__(self, processor):
        self.processor = processor

    def sales_forecaster(self):
        # Use monthly revenue series from sales_data.csv
        sales = self.processor.sales_df.copy()
        
        # Group by month
        sales['month_dt'] = sales['date'].dt.to_period('M')
        monthly = sales.groupby('month_dt')['total_revenue'].sum().reset_index()
        monthly = monthly.sort_values('month_dt')
        
        if len(monthly) == 0:
            return None
        
        # Feature: month index
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly['total_revenue'].values
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Predict next 2 months (April and May)
        X_pred = np.array([[len(monthly)], [len(monthly) + 1]])
        preds = reg.predict(X_pred)
        
        forecast = {
            'April 2026': preds[0],
            'May 2026': preds[1]
        }
        return forecast

    def churn_risk_classifier(self):
        cust = self.processor.customer_df.copy()
        
        # Engineer features
        # Mock 'today' as 2026-03-31 for the dataset context
        today = pd.to_datetime('2026-03-31')
        cust['days_since_purchase'] = (today - cust['purchase_date']).dt.days
        # Assume total_spend is purchase_amount for simplicity here based on schema
        cust['total_spend'] = cust['purchase_amount']
        cust['avg_rating'] = cust['satisfaction_rating']
        
        # Drop rows with NaN in features
        features = ['days_since_purchase', 'total_spend', 'avg_rating']
        df_clean = cust.dropna(subset=features + ['is_repeat_customer']).copy()
        
        if len(df_clean) == 0:
            return df_clean
        
        X = df_clean[features]
        # Target: is_repeat_customer (predicting True means NOT churning, False means Churn)
        # So Churn = not repeat
        y = df_clean['is_repeat_customer'].astype(int)
        
        # If there's only one class (all true or all false), we can't fit LogisticRegression easily
        if len(y.unique()) > 1:
            clf = LogisticRegression()
            clf.fit(X, y)
            
            # Predict probabilities. clf.classes_ are usually [0, 1] meaning [Churn, Repeat]
            # Probability of class 0 (Churn)
            idx_class_0 = np.where(clf.classes_ == 0)[0][0]
            probs = clf.predict_proba(X)
            churn_prob = probs[:, idx_class_0]
        else:
            churn_prob = np.where(y == 0, 1.0, 0.0)
            
        df_clean['churn_risk_score'] = churn_prob
        
        # Add risk label based on prob
        df_clean['Churn risk'] = pd.cut(df_clean['churn_risk_score'], 
                                        bins=[-1, 0.3, 0.7, 1.1], 
                                        labels=['Low', 'Medium', 'High'])
        return df_clean

    def inventory_demand_model(self):
        inv = self.processor.inventory_df.copy()
        
        # Compute a 30-day rolling average of stock_out. 
        # Wait, the inventory data as generated is just an aggregate snapshot per product?
        # Schema: product_id, product_name, category, stock_in, stock_out, current_stock, reorder_level
        # The prompt says: "compute a 30-day rolling average of stock_out per product, flag products where demand exceeds current reorder threshold."
        # Because we don't have time-series inventory, we assume `stock_out` is a total over 90 days.
        # Average per 30 days would be `stock_out` / 3.
        inv['30_day_avg_demand'] = inv['stock_out'] / 3.0
        
        # Flag where demand exceeds current reorder threshold
        inv['demand_exceeds_threshold'] = inv['30_day_avg_demand'] > inv['reorder_level']
        
        # High demand / critical restock alerts:
        flags = inv[inv['demand_exceeds_threshold']]
        return inv, flags
