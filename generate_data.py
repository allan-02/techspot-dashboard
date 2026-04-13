import os
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

def main():
    np.random.seed(42)
    random.seed(42)
    fake = Faker()
    Faker.seed(42)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1. Generate sales_data.csv
    # 247 rows: Jan–Mar 2026. Target totals: ~11,368, ~12,778 (+12.4%), ~14,363 (+12.4%) = ~38,509
    categories = ['Printers', 'Laptops', 'Networking Equipment', 'Accessories', 'Technical Services']
    products = {
        'Printers': ['LaserJet Pro', 'InkJet Color', 'Office Printer A3'],
        'Laptops': ['ThinkPad X1', 'MacBook Pro', 'Dell XPS 13', 'Budget Acer'],
        'Networking Equipment': ['Cisco Router', 'Unifi Switch', 'Mesh Wi-Fi System'],
        'Accessories': ['Wireless Mouse', 'Mechanical Keyboard', 'Monitor Stand', 'USB-C Hub'],
        'Technical Services': ['Network Setup', 'Data Recovery', 'Annual Maintenance', 'IT Consulting']
    }

    # Generate dates
    jan_dates = [fake.date_between(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 31)) for _ in range(70)]
    feb_dates = [fake.date_between(start_date=datetime(2026, 2, 1), end_date=datetime(2026, 2, 28)) for _ in range(80)]
    mar_dates = [fake.date_between(start_date=datetime(2026, 3, 1), end_date=datetime(2026, 3, 31)) for _ in range(97)]
    all_dates = sorted(jan_dates + feb_dates + mar_dates)

    # Calculate individual transaction amounts to roughly match the monthly targets
    def generate_revenues(num, target):
        dist = np.random.lognormal(mean=0, sigma=0.5, size=num)
        dist = dist / sum(dist) * target
        return np.round(dist, 2)

    jan_revs = generate_revenues(70, 11368)
    feb_revs = generate_revenues(80, 12778)
    mar_revs = generate_revenues(97, 14363)
    all_revs = np.concatenate([jan_revs, feb_revs, mar_revs])

    sales_rows = []
    for i, date in enumerate(all_dates):
        cat = random.choice(categories)
        prod = random.choice(products[cat])
        total_revenue = max(all_revs[i], 10.0) # ensure positive
        qty = random.randint(1, 5)
        # Unit price tweaked slightly to match revenue
        if cat == 'Technical Services':
            qty = 1
        unit_price = round(total_revenue / qty, 2)
        total_revenue = round(unit_price * qty, 2)
        
        sales_rows.append({
            'date': date,
            'product': prod,
            'category': cat,
            'quantity': qty,
            'unit_price': unit_price,
            'total_revenue': total_revenue
        })

    sales_df = pd.DataFrame(sales_rows)
    # Target was 38450. Adjust the last few values to hit it perfectly 
    # Current sum
    current_sum = sales_df['total_revenue'].sum()
    diff = 38450 - current_sum
    # Add diff equally to Technical Services to not mess unit prices up of hardware much
    indices = sales_df[sales_df['category'] == 'Technical Services'].index.tolist()
    if diff > 0:
        for idx in indices[:20]:
            sales_df.at[idx, 'unit_price'] += round(diff/20, 2)
            sales_df.at[idx, 'total_revenue'] = sales_df.at[idx, 'unit_price'] * sales_df.at[idx, 'quantity']
    else:
        for idx in indices[:20]:
            sales_df.at[idx, 'unit_price'] += round(diff/20, 2)
            sales_df.at[idx, 'total_revenue'] = sales_df.at[idx, 'unit_price'] * sales_df.at[idx, 'quantity']

    sales_df.to_csv(os.path.join(data_dir, 'sales_data.csv'), index=False)

    # 2. Generate customer_data.csv
    # 183 rows.
    customer_ids = [f'CUST-{1000+i}' for i in range(183)]
    ratings = []
    # Aim for avg 4.3
    for _ in range(183):
        r = np.random.choice([3, 4, 5], p=[0.1, 0.4, 0.5])
        ratings.append(r)
    
    # Adjust to exactly 4.3
    while np.mean(ratings) < 4.3:
        if 3 in ratings: ratings[ratings.index(3)] = 4
        elif 4 in ratings: ratings[ratings.index(4)] = 5
    while np.mean(ratings) > 4.3:
        if 5 in ratings: ratings[ratings.index(5)] = 4
        elif 4 in ratings: ratings[ratings.index(4)] = 3

    customer_rows = []
    for i, cid in enumerate(customer_ids):
        is_repeat = np.random.choice([True, False], p=[0.6, 0.4])
        customer_rows.append({
            'customer_id': cid,
            'name': fake.name(),
            'contact_email': fake.email(),
            'purchase_date': fake.date_between(start_date=datetime(2025, 1, 1), end_date=datetime(2026, 3, 31)),
            'purchase_amount': round(random.uniform(50, 1500), 2),
            'satisfaction_rating': ratings[i],
            'is_repeat_customer': is_repeat
        })
    pd.DataFrame(customer_rows).to_csv(os.path.join(data_dir, 'customer_data.csv'), index=False)

    # 3. Generate inventory_data.csv
    # 64 rows. Exactly 3 have current_stock < reorder_level
    inv_rows = []
    all_prods = [p for cat, lst in products.items() for p in lst if cat != 'Technical Services']
    all_prods = all_prods + [f'Other Product {i}' for i in range(64 - len(all_prods))]

    # Choose exactly 3 indices to be low stock
    low_stock_indices = random.sample(range(64), 3)

    for i in range(64):
        pname = all_prods[i]
        cat = 'Accessories' # default
        for c, p_list in products.items():
            if pname in p_list:
                cat = c
                break
        
        reorder_level = random.randint(10, 50)
        if i in low_stock_indices:
            current_stock = random.randint(0, reorder_level - 1)
        else:
            current_stock = random.randint(reorder_level, reorder_level + 100)
            
        stock_in = random.randint(50, 300)
        stock_out = random.randint(10, stock_in)

        inv_rows.append({
            'product_id': f'PRD-{100+i}',
            'product_name': pname,
            'category': cat,
            'stock_in': stock_in,
            'stock_out': stock_out,
            'current_stock': current_stock,
            'reorder_level': reorder_level
        })
    pd.DataFrame(inv_rows).to_csv(os.path.join(data_dir, 'inventory_data.csv'), index=False)

    # 4. Generate service_delivery.csv
    # 129 rows. Exactly 8 Pending
    technicians = ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince']
    service_types = ['Diagnostic', 'Repair', 'Installation', 'Consultation']

    pending_indices = set(random.sample(range(129), 8))
    serv_rows = []
    
    for i in range(129):
        status = 'Pending' if i in pending_indices else 'Completed'
        
        job_type = random.choice(service_types)
        
        # Missing rating randomly for completed, or always missing for pending
        if status == 'Pending':
            rating = np.nan
        else:
            rating = np.random.choice([1, 2, 3, 4, 5, np.nan], p=[0.05, 0.05, 0.1, 0.3, 0.4, 0.1])
            
        comp_time = np.nan if status == 'Pending' else round(random.uniform(0.5, 8.0), 1)

        serv_rows.append({
            'job_id': f'JOB-{1000+i}',
            'service_type': job_type,
            'technician_name': random.choice(technicians),
            'date': fake.date_between(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 3, 31)),
            'completion_time_hours': comp_time,
            'status': status,
            'rating': rating
        })
    
    pd.DataFrame(serv_rows).to_csv(os.path.join(data_dir, 'service_delivery.csv'), index=False)

if __name__ == '__main__':
    main()
