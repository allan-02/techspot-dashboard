import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processor import TechSightProcessor
from ml_models import TechSightML

st.set_page_config(page_title="TechSight Business Dashboard", layout="wide")

# Custom CSS for Premium Design Avoid tailwind
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4facfa;
    }
    .metric-title {
        font-size: 14px;
        color: #aaaaaa;
    }
    hr {
        margin: 1rem 0;
        border-bottom: 1px solid #333;
    }
    .alert-text {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("TechSight Business Performance Dashboard")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Navigation & Filters")

# Navigation
module_selection = st.sidebar.radio(
    "Select Module",
    ["All Modules", "Sales Insights", "Customer Analytics", "Inventory & Alerts", "Service Delivery"]
)

# Global filter
date_filter = st.sidebar.selectbox("Select Date Range", ["All Q1", "January", "February", "March"])

st.sidebar.markdown("---")
st.sidebar.header("📁 Load Datasets")
st.sidebar.caption("Optionally upload custom data. Falls back to default synthetic data if empty.")

custom_data = {}
sales_file = st.sidebar.file_uploader("Sales Data (CSV)", type="csv")
if sales_file: custom_data['sales'] = pd.read_csv(sales_file)

cust_file = st.sidebar.file_uploader("Customer Data (CSV)", type="csv")
if cust_file: custom_data['customer'] = pd.read_csv(cust_file)

inv_file = st.sidebar.file_uploader("Inventory Data (CSV)", type="csv")
if inv_file: custom_data['inventory'] = pd.read_csv(inv_file)

serv_file = st.sidebar.file_uploader("Service Delivery Data (CSV)", type="csv", key="serv")
if serv_file: custom_data['service'] = pd.read_csv(serv_file)

# We initialize processor without cache due to dynamic uploded files
try:
    processor = TechSightProcessor(custom_data=custom_data if custom_data else None)
    ml = TechSightML(processor)
except Exception as e:
    st.error(f"Error loading datasets. Please ensure they match the required schemas. Details: {e}")
    st.stop()


# Adjust compute KPIs for date filter
if date_filter == "All Q1":
    kpis = processor.compute_kpis(None)
else:
    kpis = processor.compute_kpis(date_filter)

# --- KPI HEADER ROW (Always visible) ---
st.header("KPI Insights")
c1, c2, c3, c4, c5, c6 = st.columns(6)

def render_metric(col, title, val, prefix="", suffix=""):
    col.markdown(f'''
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{prefix}{val}{suffix}</div>
    </div>
    ''', unsafe_allow_html=True)

render_metric(c1, "Total Revenue", f"{kpis['total_revenue']:,.2f}", prefix="$")
render_metric(c2, "Sales Growth", kpis['sales_growth_pct'], suffix="%")
render_metric(c3, "Satisfaction Score", kpis['satisfaction_score'], suffix="/5")
render_metric(c4, "Repeat Rate", kpis['repeat_customer_rate'], suffix="%")
render_metric(c5, "Service Completion", kpis['service_completion_rate'], suffix="%")
render_metric(c6, "Avg Completion Time", kpis['avg_completion_time'], suffix="h")

st.markdown("<br>", unsafe_allow_html=True)

# Helper config for filtering data
sales_df = processor.sales_df.copy()
cust_df = processor.customer_df.copy()
inv_df = processor.inventory_df.copy()
serv_df = processor.service_df.copy()

if date_filter != "All Q1":
    month_map = {"January": 1, "February": 2, "March": 3}
    m = month_map[date_filter]
    sales_df = sales_df[sales_df['date'].dt.month == m]
    cust_df = cust_df[cust_df['purchase_date'].dt.month == m]
    serv_df = serv_df[serv_df['date'].dt.month == m]


# --- MODULE 1: SALES ---
if module_selection in ["All Modules", "Sales Insights"]:
    with st.container():
        st.markdown("### 📊 Sales Module")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Revenue vs Target")
            
            # Monthly grouping
            monthly_rev = processor.sales_df.set_index('date').resample('M')['total_revenue'].sum().reset_index()
            monthly_rev['month_name'] = monthly_rev['date'].dt.strftime('%B')
            
            # ML Extension: Forecast
            forecast = ml.sales_forecaster()
            if forecast:
                fc_df = pd.DataFrame([
                    {'month_name': 'April', 'total_revenue': forecast['April 2026'], 'is_forecast': True},
                    {'month_name': 'May', 'total_revenue': forecast['May 2026'], 'is_forecast': True}
                ])
                monthly_rev['is_forecast'] = False
                
                fig_revenue = go.Figure()
                fig_revenue.add_trace(go.Scatter(
                    x=monthly_rev['month_name'], y=monthly_rev['total_revenue'],
                    mode='lines+markers', name='Actual', line=dict(color='#00d2ff')
                ))
                
                # Add connecting line
                if not monthly_rev.empty:
                    last_actual = monthly_rev.iloc[-1]
                    connect_fc = pd.DataFrame([last_actual, fc_df.iloc[0], fc_df.iloc[1]])
                    
                    fig_revenue.add_trace(go.Scatter(
                        x=connect_fc['month_name'], y=connect_fc['total_revenue'],
                        mode='lines+markers', name='Forecast', line=dict(dash='dash', color='purple')
                    ))
                fig_revenue.add_hline(y=5000, line_dash="dash", line_color="orange", annotation_text="Target $5,000")
            else:
                fig_revenue = px.line(monthly_rev, x='month_name', y='total_revenue', markers=True)
                fig_revenue.add_hline(y=5000, line_dash="dash", line_color="orange", annotation_text="Target $5,000")
                
            fig_revenue.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)", template="plotly_dark")
            st.plotly_chart(fig_revenue, use_container_width=True)

        with col2:
            st.subheader("Revenue by Category")
            cat_rev = sales_df.groupby('category')['total_revenue'].sum().reset_index()
            fig_cat = px.bar(cat_rev, x='category', y='total_revenue', color='category', template="plotly_dark")
            st.plotly_chart(fig_cat, use_container_width=True)

        st.subheader("Q1 Target Achievement vs $15,000 Target")
        progress = min(kpis['total_revenue'] / 15000.0, 1.0)
        st.progress(progress)
        st.caption(f"{progress * 100:.1f}% achieved of the $15,000 target. (Total Revenue: ${kpis['total_revenue']:,.2f})")
    
    if module_selection != "All Modules": st.markdown("<hr>", unsafe_allow_html=True)


# --- MODULE 2: CUSTOMER ---
if module_selection in ["All Modules", "Customer Analytics"]:
    with st.container():
        st.markdown("### 👥 Customer Module")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("New vs Repeat")
            repeat_counts = cust_df['is_repeat_customer'].value_counts().reset_index()
            repeat_counts.columns = ['Status', 'Count']
            repeat_counts['Status'] = repeat_counts['Status'].map({True: 'Repeat', False: 'New'})
            fig_pie = px.pie(repeat_counts, names='Status', values='Count', color='Status', 
                             color_discrete_map={'New': '#4facfa', 'Repeat': '#00d2ff'}, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.subheader("Monthly Target Acquisition")
            cust_acq = processor.customer_df.copy()
            cust_acq = cust_acq[~cust_acq['is_repeat_customer']]
            monthly_acq = cust_acq.set_index('purchase_date').resample('M')['customer_id'].count().reset_index()
            monthly_acq['Month'] = monthly_acq['purchase_date'].dt.strftime('%B')
            fig_acq = px.bar(monthly_acq, x='Month', y='customer_id', labels={'customer_id': 'New Customers'}, template="plotly_dark")
            st.plotly_chart(fig_acq, use_container_width=True)
            
        with c3:
            st.subheader("Satisfaction Score")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = kpis['satisfaction_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 5], 'tickwidth': 1},
                    'bar': {'color': "#00d2ff"},
                    'steps': [
                        {'range': [0, 3], 'color': "red"},
                        {'range': [3, 4], 'color': "yellow"},
                        {'range': [4, 5], 'color': "green"}],
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Customer Directory (with Churn Risk)")
        crm_df = ml.churn_risk_classifier()
        if not crm_df.empty:
            crm_display = crm_df[['customer_id', 'name', 'purchase_amount', 'satisfaction_rating', 'Churn risk']]
            
            def highlight_risk(val):
                color = 'red' if val == 'High' else 'orange' if val == 'Medium' else 'green'
                return f'color: {color}'
                
            styled_df = crm_display.style.map(highlight_risk, subset=['Churn risk'])
            st.dataframe(styled_df, use_container_width=True, height=250)
            
    if module_selection != "All Modules": st.markdown("<hr>", unsafe_allow_html=True)


# --- MODULE 3: INVENTORY ---
if module_selection in ["All Modules", "Inventory & Alerts"]:
    with st.container():
        st.markdown("### 📦 Inventory Module")
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.subheader("Current Stock vs Reorder Level")
            inv_viz = inv_df.head(15).copy()
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Bar(
                y=inv_viz['product_name'], x=inv_viz['current_stock'],
                name='Current Stock', orientation='h', marker_color='#4facfa'
            ))
            fig_inv.add_trace(go.Bar(
                y=inv_viz['product_name'], x=inv_viz['reorder_level'],
                name='Reorder Level', orientation='h', marker_color='orange'
            ))
            fig_inv.update_layout(barmode='group', template="plotly_dark", height=400)
            st.plotly_chart(fig_inv, use_container_width=True)
            
        with col_b:
            st.subheader("Alerts")
            low_stock = kpis['low_stock_items']
            if low_stock:
                st.error(f"⚠️ {len(low_stock)} items are currently below reorder levels!")
                for item in low_stock:
                    st.markdown(f"- <span class='alert-text'>{item}</span>", unsafe_allow_html=True)
            else:
                st.success("All inventory levels are optimal.")
                
            # Demand model
            try:
                inv_risk, risk_flags = ml.inventory_demand_model()
                if not risk_flags.empty:
                    st.warning("⚠️ High Demand Alerts (Rolling Forecast)")
                    for item in risk_flags['product_name'].tolist()[:5]:
                        st.markdown(f"- **{item}**: Demand exceeds regular restock.")
            except Exception as e:
                pass
                
    if module_selection != "All Modules": st.markdown("<hr>", unsafe_allow_html=True)


# --- MODULE 4: SERVICE DELIVERY ---
if module_selection in ["All Modules", "Service Delivery"]:
    with st.container():
        st.markdown("### 🔧 Service Delivery Module")
        s1, s2 = st.columns([1, 1])
        
        with s1:
            st.subheader("Jobs per Technician")
            tech_counts = serv_df.groupby(['technician_name', 'status']).size().reset_index(name='Count')
            if not tech_counts.empty:
                fig_serv = px.bar(tech_counts, x='technician_name', y='Count', color='status',
                                  color_discrete_map={'Completed': '#00d2ff', 'Pending': 'orange'}, 
                                  template="plotly_dark")
                st.plotly_chart(fig_serv, use_container_width=True)
            
        with s2:
            st.subheader("Technician Leaderboard")
            board = kpis.get('technician_leaderboard', pd.DataFrame())
            if not board.empty:
                def get_badge(r):
                    if r >= 4.5: return '🏆 Elite'
                    elif r >= 3.5: return '⭐ Good'
                    else: return '⚠️ Needs Impr.'
                    
                board['status_badge'] = board['avg_rating'].apply(get_badge)
                st.dataframe(board[['technician_name', 'avg_rating', 'job_count', 'status_badge']], 
                             use_container_width=True, hide_index=True)
