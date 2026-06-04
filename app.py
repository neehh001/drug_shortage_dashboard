import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import pickle
from xgboost import plot_importance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- Page config ---
st.set_page_config(
    page_title="Drug Shortage Intelligence",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Hide Streamlit branding + custom CSS ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #0a0e1a;
}

[data-testid="block-container"] {
    padding: 2rem 3rem;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem;
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700;
    color: #f1f5f9;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Divider */
hr {
    border-color: #1e293b;
}

/* Subheader */
h2, h3 {
    color: #f1f5f9;
    font-weight: 600;
}

/* Selectbox and slider */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] > div {
    background: #111827;
    border-radius: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 12px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0e1a;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}

/* Tag pills */
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.tag-red { background: #450a0a; color: #fca5a5; }
.tag-yellow { background: #422006; color: #fcd34d; }
.tag-green { background: #052e16; color: #86efac; }
</style>
""", unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
    conn = sqlite3.connect('drug_shortages.db')
    df = pd.read_sql("SELECT * FROM shortages_clean", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# --- Chart config ---
CHART_BG = '#0a0e1a'
GRID_COLOR = '#1e293b'
TEXT_COLOR = '#94a3b8'
RED = '#ef4444'
ORANGE = '#f97316'
SLATE = '#334155'

def style_fig(fig):
    fig.update_layout(
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR, family='Inter'),
        xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_COLOR))
    )
    return fig

# --- Header ---
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='font-size: 1.8rem; font-weight: 700; color: #f1f5f9; margin: 0;'>
        ⚕ Drug Shortage Intelligence
    </h1>
    <p style='color: #64748b; margin: 0.3rem 0 0 0; font-size: 0.9rem;'>
        FDA shortage data · 1,673 records · Updated from public registry
    </p>
</div>
""", unsafe_allow_html=True)

# --- KPI Cards ---
active = df[df['Status'] == 'Current']
critical = df[(df['Status'] == 'Current') & (df['is_critical'] == 1)]
avg_duration = int(df['shortage_duration_days'].mean())
high_risk = df[df['severity_score'] >= 70]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Shortages", f"{len(active):,}")
col2.metric("Critical Categories", f"{len(critical):,}")
col3.metric("Avg Duration", f"{avg_duration:,} days")
col4.metric("High Risk (≥70)", f"{len(high_risk):,}")

st.markdown("<hr>", unsafe_allow_html=True)

# --- Row 1: Charts ---
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### Shortages by Category")
    cat_counts = (
        active.groupby('Therapeutic Category')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=True)
        .tail(12)
    )
    fig1 = px.bar(
        cat_counts,
        x='count',
        y='Therapeutic Category',
        orientation='h',
        color='count',
        color_continuous_scale=[[0, '#1e293b'], [0.5, '#b91c1c'], [1, '#ef4444']],
    )
    fig1.update_traces(marker_line_width=0)
    fig1.update_coloraxes(showscale=False)
    fig1 = style_fig(fig1)
    fig1.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("#### Status Breakdown")
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig2 = px.pie(
        status_counts,
        values='Count',
        names='Status',
        color_discrete_sequence=[RED, ORANGE, '#22c55e'],
        hole=0.6
    )
    fig2.update_traces(
        textinfo='percent',
        textfont=dict(color='white', size=12),
        marker=dict(line=dict(color=CHART_BG, width=2))
    )
    fig2 = style_fig(fig2)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Row 2: Scatter ---
st.markdown("#### Severity vs Duration")
fig3 = px.scatter(
    active,
    x='shortage_duration_days',
    y='severity_score',
    color='is_critical',
    color_discrete_map={0: SLATE, 1: RED},
    hover_data=['Generic Name', 'Therapeutic Category'],
    labels={
        'shortage_duration_days': 'Duration (Days)',
        'severity_score': 'Severity Score',
        'is_critical': 'Critical'
    },
    opacity=0.75
)
fig3.add_hline(
    y=70,
    line_dash="dot",
    line_color="#ef4444",
    line_width=1,
    annotation_text="High Risk Threshold",
    annotation_font_color="#ef4444",
    annotation_font_size=11
)
fig3 = style_fig(fig3)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Row 3: Risk Table ---
st.markdown("#### Shortage Risk Table")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    status_filter = st.selectbox("Status", ['All'] + df['Status'].unique().tolist())
with col_f2:
    critical_filter = st.selectbox("Type", ['All', 'Critical Only', 'Non-Critical'])
with col_f3:
    min_score = st.slider("Min Severity Score", 0, 100, 0)

filtered = df.copy()
if status_filter != 'All':
    filtered = filtered[filtered['Status'] == status_filter]
if critical_filter == 'Critical Only':
    filtered = filtered[filtered['is_critical'] == 1]
elif critical_filter == 'Non-Critical':
    filtered = filtered[filtered['is_critical'] == 0]
filtered = filtered[filtered['severity_score'] >= min_score]

def color_severity(val):
    if val >= 70:
        return 'background-color: #450a0a; color: #fca5a5'
    elif val >= 40:
        return 'background-color: #422006; color: #fcd34d'
    else:
        return 'background-color: #052e16; color: #86efac'

display_cols = ['Generic Name', 'Therapeutic Category',
                'Company Name', 'Status',
                'shortage_duration_days', 'severity_score']

st.dataframe(
    filtered[display_cols]
    .sort_values('severity_score', ascending=False)
    .style.map(color_severity, subset=['severity_score']),
    use_container_width=True,
    height=380
)

st.caption(f"{len(filtered):,} of {len(df):,} records shown")

st.markdown("<hr>", unsafe_allow_html=True)

# --- Row 4: Drug Deep Dive ---
st.markdown("#### Drug Deep Dive")

selected_drug = st.selectbox(
    "Select a drug",
    df['Generic Name'].sort_values().unique(),
    key='drug_select'
)

drug_data = df[df['Generic Name'] == selected_drug]

if not drug_data.empty:
    row = drug_data.iloc[0]

    # Risk tag
    score = row['severity_score']
    if score >= 70:
        tag = '<span class="tag tag-red">High Risk</span>'
    elif score >= 40:
        tag = '<span class="tag tag-yellow">Medium Risk</span>'
    else:
        tag = '<span class="tag tag-green">Low Risk</span>'

    st.markdown(f"""
    <div style='background:#111827; border:1px solid #1e293b; border-radius:12px; padding:1.5rem; margin-bottom:1rem;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;'>
            <h3 style='margin:0; color:#f1f5f9;'>{row['Generic Name']}</h3>
            {tag}
        </div>
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; color:#94a3b8; font-size:0.85rem;'>
            <div><span style='color:#64748b;'>Company</span><br><span style='color:#e2e8f0;'>{row['Company Name']}</span></div>
            <div><span style='color:#64748b;'>Category</span><br><span style='color:#e2e8f0;'>{row['Therapeutic Category']}</span></div>
            <div><span style='color:#64748b;'>Status</span><br><span style='color:#e2e8f0;'>{row['Status']}</span></div>
            <div><span style='color:#64748b;'>Duration</span><br><span style='color:#e2e8f0;'>{int(row['shortage_duration_days'])} days</span></div>
            <div><span style='color:#64748b;'>Severity Score</span><br><span style='color:#e2e8f0;'>{row['severity_score']}</span></div>
            <div><span style='color:#64748b;'>Critical Category</span><br><span style='color:#e2e8f0;'>{'Yes' if row['is_critical'] else 'No'}</span></div>
        </div>
        <div style='margin-top:1rem; color:#94a3b8; font-size:0.85rem;'>
            <span style='color:#64748b;'>Availability</span><br>
            <span style='color:#e2e8f0;'>{row['Availability Information']}</span>
        </div>
        <div style='margin-top:0.8rem; color:#94a3b8; font-size:0.85rem;'>
            <span style='color:#64748b;'>Reason for Shortage</span><br>
            <span style='color:#e2e8f0;'>{row['Reason for Shortage']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'color': '#f1f5f9', 'size': 40}},
        gauge={
            'axis': {'range': [0, 100],
                     'tickcolor': TEXT_COLOR,
                     'tickfont': {'color': TEXT_COLOR}},
            'bar': {'color': RED if score >= 70 else ORANGE if score >= 40 else '#22c55e'},
            'bgcolor': CHART_BG,
            'bordercolor': GRID_COLOR,
            'steps': [
                {'range': [0, 40], 'color': '#052e16'},
                {'range': [40, 70], 'color': '#422006'},
                {'range': [70, 100], 'color': '#450a0a'}
            ],
        }
    ))
    fig4.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=30, r=30, t=30, b=10),
        height=250
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ML Prediction
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### ML Prediction — Will This Shortage Exceed 1 Year?")

    drug_features = pd.DataFrame([{
        'is_critical': row['is_critical'],
        'no_availability': row['no_availability'],
        'duration_normalized': row['duration_normalized'],
        'is_active': row['is_active']
    }])

    prob = model.predict_proba(drug_features)[0][1]

    col_p1, col_p2 = st.columns(2)
    col_p1.metric(
        "Prediction",
        "High Risk" if prob > 0.5 else "Lower Risk"
    )
    col_p2.metric(
        "Probability of Long Shortage",
        f"{prob:.1%}"
    )

st.markdown("<hr>", unsafe_allow_html=True)

# --- Feature Importance ---
st.markdown("#### Model — Feature Importance")
fig5, ax = plt.subplots(figsize=(8, 3))
fig5.patch.set_facecolor(CHART_BG)
ax.set_facecolor(CHART_BG)
plot_importance(
    model, ax=ax,
    color='#ef4444',
    grid=False,
    show_values=False,
    importance_type='gain'
)
ax.tick_params(colors=TEXT_COLOR)
ax.xaxis.label.set_color(TEXT_COLOR)
ax.yaxis.label.set_color(TEXT_COLOR)
ax.title.set_color(TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)
plt.tight_layout()
st.pyplot(fig5)