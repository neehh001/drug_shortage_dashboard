import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(
    page_title="Drug Shortage Early Warning System",
    page_icon="💊",
    layout="wide"
)

# --- Load data ---
@st.cache_data
def load_data():
    conn = sqlite3.connect('drug_shortages.db')
    df = pd.read_sql("SELECT * FROM shortages_clean", conn)
    conn.close()
    return df

df = load_data()

# --- Header ---
st.title("Drug Shortage Early Warning System")
st.markdown("Real-time FDA drug shortage intelligence for procurement teams")
st.divider()

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)

active = df[df['Status'] == 'Current']
critical = df[(df['Status'] == 'Current') & (df['is_critical'] == 1)]
avg_duration = int(df['shortage_duration_days'].mean())
high_risk = df[df['severity_score'] >= 70]

col1.metric("Active Shortages", len(active))
col2.metric("Critical Category Shortages", len(critical))
col3.metric("Avg Shortage Duration", f"{avg_duration} days")
col4.metric("High Risk (Score ≥ 70)", len(high_risk))

st.divider()

# --- Row 1: Two charts side by side ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Shortages by Therapeutic Category")
    cat_counts = (
        active.groupby('Therapeutic Category')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(10)
    )
    fig1 = px.bar(
        cat_counts,
        x='count', y='Therapeutic Category',
        orientation='h',
        color='count',
        color_continuous_scale='Reds',
        labels={'count': 'Number of Shortages'}
    )
    fig1.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("Shortage Status Breakdown")
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig2 = px.pie(
        status_counts,
        values='Count',
        names='Status',
        color_discrete_sequence=['#ef4444', '#f97316', '#22c55e']
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Row 2: Severity scatter ---
st.subheader("Severity vs Duration — Active Shortages")

fig3 = px.scatter(
    active,
    x='shortage_duration_days',
    y='severity_score',
    color='Therapeutic Category',
    hover_data=['Generic Name', 'Company Name', 'Status'],
    labels={
        'shortage_duration_days': 'Shortage Duration (Days)',
        'severity_score': 'Severity Score'
    },
    opacity=0.7
)
fig3.add_hline(y=70, line_dash="dash", line_color="red",
               annotation_text="High Risk Threshold (70)")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Row 3: Risk table with filters ---
st.subheader("Drug Shortage Risk Table")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    status_filter = st.selectbox("Filter by Status", 
                                  ['All'] + df['Status'].unique().tolist())
with col_f2:
    critical_filter = st.selectbox("Filter by Type", 
                                    ['All', 'Critical Only', 'Non-Critical'])
with col_f3:
    min_score = st.slider("Minimum Severity Score", 0, 100, 0)

# Apply filters
filtered = df.copy()
if status_filter != 'All':
    filtered = filtered[filtered['Status'] == status_filter]
if critical_filter == 'Critical Only':
    filtered = filtered[filtered['is_critical'] == 1]
elif critical_filter == 'Non-Critical':
    filtered = filtered[filtered['is_critical'] == 0]
filtered = filtered[filtered['severity_score'] >= min_score]

# Color code severity
def color_severity(val):
    if val >= 70:
        return 'background-color: #fee2e2'
    elif val >= 40:
        return 'background-color: #fef9c3'
    else:
        return 'background-color: #dcfce7'

display_cols = ['Generic Name', 'Therapeutic Category', 'Company Name',
                'Status', 'shortage_duration_days', 'severity_score']

st.dataframe(
    filtered[display_cols]
    .sort_values('severity_score', ascending=False)
    .style.map(color_severity, subset=['severity_score']),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(filtered)} of {len(df)} records")

st.divider()

# --- Row 4: Drug deep dive ---
st.subheader("Drug Deep Dive")

selected_drug = st.selectbox(
    "Select a drug to inspect",
    df['Generic Name'].sort_values().unique()
)

drug_data = df[df['Generic Name'] == selected_drug]

if not drug_data.empty:
    row = drug_data.iloc[0]
    
    d1, d2, d3 = st.columns(3)
    d1.metric("Severity Score", row['severity_score'])
    d2.metric("Duration (Days)", int(row['shortage_duration_days']))
    d3.metric("Status", row['Status'])
    
    st.markdown(f"**Company:** {row['Company Name']}")
    st.markdown(f"**Category:** {row['Therapeutic Category']}")
    st.markdown(f"**Availability Info:** {row['Availability Information']}")
    st.markdown(f"**Reason for Shortage:** {row['Reason for Shortage']}")

    # Severity gauge
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row['severity_score'],
        title={'text': "Severity Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "#dcfce7"},
                {'range': [40, 70], 'color': "#fef9c3"},
                {'range': [70, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    st.plotly_chart(fig4, use_container_width=True)


    st.divider()

# --- ML Model Section ---
st.subheader("🤖 Predictive Model — Will This Shortage Last Over 1 Year?")

st.markdown("""
This XGBoost model predicts whether an active shortage will exceed 
365 days based on category criticality, availability status, and duration.
""")

# Feature importance chart
from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
plot_importance(model, ax=ax, 
                title='Feature Importance — Drug Shortage Predictor')
plt.tight_layout()
st.pyplot(fig)

# Per drug prediction
st.subheader("Shortage Duration Prediction")
selected = st.selectbox(
    "Select a drug to predict",
    df['Generic Name'].sort_values().unique(),
    key='ml_select'
)

drug_row = df[df['Generic Name'] == selected].iloc[0]
drug_features = pd.DataFrame([{
    'is_critical': drug_row['is_critical'],
    'no_availability': drug_row['no_availability'],
    'duration_normalized': drug_row['duration_normalized'],
    'is_active': drug_row['is_active']
}])

prob = model.predict_proba(drug_features)[0][1]
prediction = "🔴 High Risk — Likely to exceed 1 year" if prob > 0.5 else "🟢 Lower Risk"

st.metric("Prediction", prediction)
st.metric("Probability of Long Shortage", f"{prob:.1%}")