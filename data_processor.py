import os
import pandas as pd
import numpy as np

class TechSightProcessor:
    def __init__(self, data_dir=None, custom_data=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        self.custom_data = custom_data or {}
        self.sales_df = None
        self.customer_df = None
        self.inventory_df = None
        self.service_df = None
        self.load_and_clean_data()

    def load_and_clean_data(self):
        # 1. Load data
        if 'sales' in self.custom_data:
            self.sales_df = self.custom_data['sales'].copy()
        else:
            self.sales_df = pd.read_csv(os.path.join(self.data_dir, 'sales_data.csv'))

        if 'customer' in self.custom_data:
            self.customer_df = self.custom_data['customer'].copy()
        else:
            self.customer_df = pd.read_csv(os.path.join(self.data_dir, 'customer_data.csv'))

        if 'inventory' in self.custom_data:
            self.inventory_df = self.custom_data['inventory'].copy()
        else:
            self.inventory_df = pd.read_csv(os.path.join(self.data_dir, 'inventory_data.csv'))

        if 'service' in self.custom_data:
            self.service_df = self.custom_data['service'].copy()
        else:
            self.service_df = pd.read_csv(os.path.join(self.data_dir, 'service_delivery.csv'))

        # 2. Clean data
        # Drop duplicates
        self.sales_df.drop_duplicates(inplace=True)
        self.customer_df.drop_duplicates(inplace=True)
        self.inventory_df.drop_duplicates(inplace=True)
        self.service_df.drop_duplicates(inplace=True)

        # Parse dates
        self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
        self.customer_df['purchase_date'] = pd.to_datetime(self.customer_df['purchase_date'])
        self.service_df['date'] = pd.to_datetime(self.service_df['date'])

        # Validate numeric columns
        numeric_cols_sales = ['quantity', 'unit_price', 'total_revenue']
        for col in numeric_cols_sales:
            self.sales_df[col] = pd.to_numeric(self.sales_df[col], errors='coerce').fillna(0)

        self.customer_df['purchase_amount'] = pd.to_numeric(self.customer_df['purchase_amount'], errors='coerce').fillna(0)
        self.customer_df['satisfaction_rating'] = pd.to_numeric(self.customer_df['satisfaction_rating'], errors='coerce')

        inv_num_cols = ['stock_in', 'stock_out', 'current_stock', 'reorder_level']
        for col in inv_num_cols:
            self.inventory_df[col] = pd.to_numeric(self.inventory_df[col], errors='coerce').fillna(0)

        self.service_df['completion_time_hours'] = pd.to_numeric(self.service_df['completion_time_hours'], errors='coerce')
        self.service_df['rating'] = pd.to_numeric(self.service_df['rating'], errors='coerce')

        # Fill missing ratings in service_delivery with per-service_type median
        # For pending jobs, rating shouldn't artificially boost, but assignment says:
        # "fills missing rating values with per-service_type median". We will follow instructions carefully.
        self.service_df['rating'] = self.service_df.groupby('service_type')['rating'].transform(lambda x: x.fillna(x.median()))
        
        # Fallback if any median was NaN (all missing for a group)
        overall_median = self.service_df['rating'].median()
        self.service_df['rating'] = self.service_df['rating'].fillna(overall_median)

        # Ensure no nulls are left in the entire dataframes
        self.sales_df = self.sales_df.fillna(0)
        self.customer_df = self.customer_df.fillna(0)
        self.inventory_df = self.inventory_df.fillna(0)
        self.service_df = self.service_df.fillna(0)

    def compute_kpis(self, date_range=None):
        sales = self.sales_df.copy()
        cust = self.customer_df.copy()
        inv = self.inventory_df.copy()
        serv = self.service_df.copy()

        # Date range filtering if needed (for Streamlit integration)
        if date_range == "January":
            sales = sales[sales['date'].dt.month == 1]
            serv = serv[serv['date'].dt.month == 1]
        elif date_range == "February":
            sales = sales[sales['date'].dt.month == 2]
            serv = serv[serv['date'].dt.month == 2]
        elif date_range == "March":
            sales = sales[sales['date'].dt.month == 3]
            serv = serv[serv['date'].dt.month == 3]

        # total_revenue
        total_revenue = sales['total_revenue'].sum()

        # sales_growth_pct - generally calculated on full dataset as MoM (Mar vs Feb)
        # If user filters, we still return the overall dataset's Mar vs Feb growth or calculate based on the current context
        # Given it's a fixed +12.4% MoM goal, let's use the full dataset for this metric to stay accurate.
        monthly_rev = self.sales_df.groupby(self.sales_df['date'].dt.to_period('M'))['total_revenue'].sum()
        if len(monthly_rev) >= 2:
            last_m = monthly_rev.iloc[-1]
            prev_m = monthly_rev.iloc[-2]
            sales_growth_pct = ((last_m - prev_m) / prev_m) * 100
        else:
            sales_growth_pct = 0.0

        # top_product
        top_product = sales.groupby('product')['total_revenue'].sum().idxmax() if not sales.empty else "N/A"

        # total_customers
        total_customers = cust['customer_id'].nunique()

        # new_customers_q1
        # Q1 2026 means Jan-Mar 2026. 
        # Typically "acquired" means their purchase was in Q1 2026 and they are not repeat customers.
        q1_mask = (cust['purchase_date'] >= '2026-01-01') & (cust['purchase_date'] <= '2026-03-31')
        new_mask = ~cust['is_repeat_customer']
        new_customers_q1 = cust[q1_mask & new_mask]['customer_id'].nunique()

        # satisfaction_score
        satisfaction_score = round(cust['satisfaction_rating'].mean(), 1)

        # repeat_customer_rate
        repeat_rate_val = cust['is_repeat_customer'].mean() * 100
        repeat_customer_rate = round(repeat_rate_val, 1) if not np.isnan(repeat_rate_val) else 0.0

        # low_stock_items
        low_stock_items = inv[inv['current_stock'] < inv['reorder_level']]['product_name'].tolist()

        # services_completed
        services_completed = len(serv[serv['status'] == 'Completed'])

        # service_completion_rate
        total_serv = len(serv)
        service_completion_rate = (services_completed / total_serv * 100) if total_serv > 0 else 0.0

        # avg_completion_time
        avg_completion_time = serv['completion_time_hours'].mean()

        # technician_leaderboard
        lead = serv.groupby('technician_name').agg({
            'rating': 'mean',
            'job_id': 'count'
        }).rename(columns={'job_id': 'job_count', 'rating': 'avg_rating'}).reset_index()
        technician_leaderboard = lead.sort_values(by='job_count', ascending=False)
        technician_leaderboard['avg_rating'] = technician_leaderboard['avg_rating'].round(1)

        return {
            'total_revenue': total_revenue,
            'sales_growth_pct': float(round(sales_growth_pct, 1)),
            'top_product': top_product,
            'total_customers': int(total_customers),
            'new_customers_q1': int(new_customers_q1),
            'satisfaction_score': satisfaction_score,
            'repeat_customer_rate': repeat_customer_rate,
            'low_stock_items': low_stock_items,
            'services_completed': int(services_completed),
            'service_completion_rate': float(round(service_completion_rate, 1)),
            'avg_completion_time': float(round(avg_completion_time, 1)),
            'technician_leaderboard': technician_leaderboard
        }
