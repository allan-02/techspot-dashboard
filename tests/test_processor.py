import pytest
import os
import pandas as pd
from data_processor import TechSightProcessor

@pytest.fixture
def processor():
    return TechSightProcessor()

def test_total_revenue(processor):
    kpis = processor.compute_kpis()
    csv_df = pd.read_csv(os.path.join(processor.data_dir, 'sales_data.csv'))
    expected_rev = pd.to_numeric(csv_df['total_revenue'], errors='coerce').fillna(0).sum()
    
    assert abs(kpis['total_revenue'] - expected_rev) < 0.01

def test_low_stock_items_returns_correct_products(processor):
    kpis = processor.compute_kpis()
    
    csv_df = pd.read_csv(os.path.join(processor.data_dir, 'inventory_data.csv'))
    csv_df['current_stock'] = pd.to_numeric(csv_df['current_stock'], errors='coerce').fillna(0)
    csv_df['reorder_level'] = pd.to_numeric(csv_df['reorder_level'], errors='coerce').fillna(0)
    
    expected_items = csv_df[csv_df['current_stock'] < csv_df['reorder_level']]['product_name'].tolist()
    
    assert sorted(kpis['low_stock_items']) == sorted(expected_items)
    # the prompt specifies "Ensure 3 products have current_stock below reorder_level"
    assert len(kpis['low_stock_items']) == 3

def test_satisfaction_score(processor):
    kpis = processor.compute_kpis()
    csv_df = pd.read_csv(os.path.join(processor.data_dir, 'customer_data.csv'))
    expected_score = round(pd.to_numeric(csv_df['satisfaction_rating'], errors='coerce').mean(), 1)
    
    assert kpis['satisfaction_score'] == expected_score

def test_service_completion_rate(processor):
    kpis = processor.compute_kpis()
    
    rate = kpis['service_completion_rate']
    assert 0.0 <= rate <= 100.0

def test_no_null_values_in_cleaned_data(processor):
    assert not processor.sales_df.isnull().values.any()
    assert not processor.customer_df.isnull().values.any()
    assert not processor.inventory_df.isnull().values.any()
    assert not processor.service_df.isnull().values.any()
