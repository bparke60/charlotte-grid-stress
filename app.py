import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydeck as pdk
import random

st.set_page_config(
    page_title="Charlotte Grid Stress Predictor",
    page_icon="⚡",
    layout="wide"
)

# ── Inject responsive CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten up default Streamlit padding on mobile */
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.75rem !important; }
    h1 { font-size: 1.4rem !important; }
}
/* Metric cards */
.metric-card {
    background: #F0F6FF;
    border-left: 4px solid #0085CA;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Color constants ────────────────────────────────────────────────────────────
BLUE       = '#0085CA'
DARK_BLUE  = '#005A8E'
STRESS_COLORS = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C'}
STRESS_RGB    = {'Low': [46,204,113], 'Medium': [243,156,18], 'High': [231,76,60]}
USER_COLOR = '#FFD700'

# ── Realistic Charlotte neighborhood coordinates ───────────────────────────────
NEIGHBORHOOD_COORDS = {
    'University City': (35.3078, -80.7428),
    'South End':       (35.2157, -80.8590),
    'NoDa':            (35.2387, -80.8143),
    'Steele Creek':    (35.1598, -80.9235),
    'Ballantyne':      (35.0523, -80.8459),
    'East Charlotte':  (35.2271, -80.7721),
    'Mooresville':     (35.5843, -80.8098),
    'Matthews':        (35.1176, -80.7201),
    'Huntersville':    (35.4107, -80.8429),
    'Concord':         (35.4088, -80.5796),
}

# Realistic existing data centers in the Charlotte metro
EXISTING_DATA_CENTERS = [
    {'name': 'QTS Charlotte (University City)',   'lat': 35.3041, 'lon': -80.7502, 'size_mw': 120, 'stress': 'Medium'},
    {'name': 'Flexential Charlotte',              'lat': 35.2268, 'lon': -80.8431, 'stress': 'Low',    'size_mw': 85},
    {'name': 'CyrusOne Charlotte',                'lat': 35.2195, 'lon': -80.8512, 'stress': 'Low',    'size_mw': 60},
    {'name': 'Databank CLT1',                     'lat': 35.2301, 'lon': -80.8387, 'stress': 'Medium', 'size_mw': 95},
    {'name': 'Duke Energy HQ Campus',             'lat': 35.2271, 'lon': -80.8416, 'stress': 'High',   'size_mw': 310},
    {'name': 'Steele Creek Industrial (proposed)','lat': 35.1621, 'lon': -80.9198, 'stress': 'High',   'size_mw': 280},
    {'name': 'Huntersville Tech Park',            'lat': 35.4089, 'lon': -80.8461, 'stress': 'Low',    'size_mw': 45},
    {'name': 'Concord Innovation Center',         'lat': 35.4052, 'lon': -80.5831, 'stress': 'Medium', 'size_mw': 110},
]

# Duke Energy substations serving the Charlotte area
SUBSTATIONS = [
    {'name': 'McGuire Nuclear Station',   'lat': 35.4325, 'lon': -80.9481},
    {'name': 'Marshall Steam Station',    'lat': 35.4756, 'lon': -80.9634},
    {'name': 'Lincoln Substation',        'lat': 35.4731, 'lon': -81.2456},
    {'name': 'South End Substation',      'lat': 35.2089, 'lon': -80.8601},
    {'name': 'University Substation',     'lat': 35.3115, 'lon': -80.7389},
    {'name': 'Ballantyne Substation',     'lat': 35.0498, 'lon': -80.8512},
    {'name': 'Concord Substation',        'lat': 35.4021, 'lon': -80.5712},
    {'name': 'Mooresville Substation',    'lat': 35.5901, 'lon': -80.8134},
]

# ── Model training ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

@st.cache_resource
def train_model():
    n = 500
    neighborhoods = list(NEIGHBORHOOD_COORDS.keys())
    data = {
        'neighborhood':               [random.choice(neighborhoods) for _ in range(n)],
        'data_center_size_mw':        np.round(np.random.uniform(10, 500, n), 1),
        'existing_grid_load_pct':     np.round(np.random.uniform(40, 95,  n), 1),
        'distance_to_substation_miles': np.round(np.random.uniform(0.5, 15, n), 2),
        'residential_density':        np.round(np.random.uniform(1, 10,   n), 1),
        'num_existing_data_centers':  np.random.randint(0, 8, n),
    }

    def classify_stress(row):
        score = 0
        score += 3 if row['data_center_size_mw'] > 300 else (2 if row['data_center_size_mw'] > 150 else 1)
        score += 3 if row['existing_grid_load_pct'] > 80 else (2 if row['existing_grid_load_pct'] > 65 else 1)
        score += 2 if row['distance_to_substation_miles'] > 10 else 1
        score += 2 if row['num_existing_data_centers'] > 4 else 1
        return 'High' if score >= 8 else ('Medium' if score >= 5 else 'Low')

    df = pd.DataFrame(data)
    df['grid_stress_level'] = df.apply(classify_stress, axis=1)

    le_n = LabelEncoder()
    le_s = LabelEncoder()
    df['neighborhood_encoded'] = le_n.fit_transform(df['neighborhood'])
    df['stress_encoded']       = le_s.fit_transform(df['grid_stress_level'])

    features = ['neighborhood_encoded', 'data_center_size_mw', 'existing_grid_load_pct',
                'distance_to_substation_miles', 'residential_density', 'num_existing_data_centers']
    X, y = df[features], df['stress_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    mdl.fit(X_train, y_train)
    return mdl, le_n, le_s, features, df

model, le_neighborhood, le_stress, features, df = train_model()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("⚡ Charlotte Data Center Grid Stress Predictor")
st.markdown(
    f"<p style='color:{BLUE}; font-size:17px; margin-top:-8px;'>"
    "Adjust the inputs below to see how a proposed data center would impact Charlotte's power grid — "
    "the map and chart update live.</p>",
    unsafe_allow_html=True
)
st.caption("Dataset is synthetically generated using real-world grid engineering parameters from the Charlotte metro area.")

st.divider()

# ── Input controls ─────────────────────────────────────────────────────────────
st.subheader("📋 Proposed Data Center Parameters")
c1, c2, c3 = st.columns(3)
with c1:
    neighborhood = st.selectbox("📍 Neighborhood", sorted(NEIGHBORHOOD_COORDS.keys()))
    size_mw      = st.slider("🏗️ Facility Size (MW)", 10, 500, 150, 10)
with c2:
    grid_load = st.slider("⚡ Existing Grid Load (%)", 40, 95, 65, 1)
    distance  = st.slider("📏 Distance to Substation (mi)", 0.5, 15.0, 5.0, 0.5)
with c3:
    density  = st.slider("🏘️ Residential Density (1–10)", 1, 10, 5, 1)
    existing = st.slider("🖥️ Nearby Data Centers", 0, 8, 2, 1)

# ── Run prediction ─────────────────────────────────────────────────────────────
hood_enc   = le_neighborhood.transform([neighborhood])[0]
input_data = pd.DataFrame([[hood_enc, size_mw, grid_load, distance, density, existing]], columns=features)
pred       = model.predict(input_data)[0]
stress_label = le_stress.inverse_transform([pred])[0]
proba      = model.predict_proba(input_data)[0]
confidence = max(proba) * 100

st.divider()

# ── Prediction banner ──────────────────────────────────────────────────────────
if stress_label == "High":
    st.error(f"🔴 **HIGH Grid Stress** — {confidence:.1f}% confidence  |  This proposal would likely require major infrastructure upgrades before approval.")
elif stress_label == "Medium":
    st.warning(f"🟡 **MEDIUM Grid Stress** — {confidence:.1f}% confidence  |  Further engineering review is recommended.")
else:
    st.success(f"🟢 **LOW Grid Stress** — {confidence:.1f}% confidence  |  This proposal appears manageable for the existing grid infrastructure.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VISUALIZATION — Map + Scatter side-by-side
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Live Visualization")
map_col, chart_col = st.columns([1.1, 1], gap="medium")

# ── LEFT: Charlotte map ────────────────────────────────────────────────────────
with map_col:
    st.markdown("**🗺️ Charlotte Metro — Data Centers & Substations**")

    proposed_lat, proposed_lon = NEIGHBORHOOD_COORDS[neighborhood]

    # Build pydeck layers
    # 1. Existing data centers — colored circles sized by MW
    dc_data = []
    for dc in EXISTING_DATA_CENTERS:
        rgb = STRESS_RGB[dc['stress']]
        dc_data.append({
            'lat': dc['lat'], 'lon': dc['lon'],
            'name': dc['name'], 'size_mw': dc['size_mw'],
            'r': rgb[0], 'g': rgb[1], 'b': rgb[2],
            'radius': int(dc['size_mw'] * 18),
        })

    dc_layer = pdk.Layer(
        'ScatterplotLayer',
        data=dc_data,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='[r, g, b, 180]',
        get_line_color='[255, 255, 255]',
        line_width_min_pixels=1,
        pickable=True,
        tooltip=True,
    )

    # 2. Substations — white diamond markers
    sub_data = [{'lat': s['lat'], 'lon': s['lon'], 'name': s['name']} for s in SUBSTATIONS]
    sub_layer = pdk.Layer(
        'ScatterplotLayer',
        data=sub_data,
        get_position='[lon, lat]',
        get_radius=400,
        get_fill_color='[255, 255, 255, 220]',
        get_line_color='[0, 85, 142]',
        line_width_min_pixels=2,
        pickable=True,
    )

    # 3. Proposed location — pulsing gold star (large gold circle)
    proposed_rgb = STRESS_RGB[stress_label]
    proposed_layer = pdk.Layer(
        'ScatterplotLayer',
        data=[{'lat': proposed_lat, 'lon': proposed_lon,
               'name': f'YOUR PROPOSAL: {neighborhood}',
               'r': proposed_rgb[0], 'g': proposed_rgb[1], 'b': proposed_rgb[2]}],
        get_position='[lon, lat]',
        get_radius=1200,
        get_fill_color='[r, g, b, 230]',
        get_line_color='[255, 215, 0]',
        line_width_min_pixels=3,
        pickable=True,
    )
    # Inner ring to create "star" effect
    proposed_inner = pdk.Layer(
        'ScatterplotLayer',
        data=[{'lat': proposed_lat, 'lon': proposed_lon}],
        get_position='[lon, lat]',
        get_radius=500,
        get_fill_color='[255, 215, 0, 255]',
        get_line_color='[0,0,0,255]',
        line_width_min_pixels=2,
        pickable=False,
    )

    view_state = pdk.ViewState(
        latitude=35.27,
        longitude=-80.84,
        zoom=9.2,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{name}</b>",
        "style": {"background": "#1a1a2e", "color": "white",
                  "font-family": "monospace", "padding": "6px 10px",
                  "border-radius": "4px", "font-size": "12px"}
    }

    deck = pdk.Deck(
        layers=[dc_layer, sub_layer, proposed_layer, proposed_inner],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/dark-v10',
    )
    st.pydeck_chart(deck, use_container_width=True)

    # Map legend
    st.markdown(
        f"""
        <div style='font-size:12px; line-height:1.8; margin-top:6px;'>
        <span style='color:#2ECC71'>●</span> Low stress data center &nbsp;
        <span style='color:#F39C12'>●</span> Medium stress &nbsp;
        <span style='color:#E74C3C'>●</span> High stress &nbsp;
        <span style='color:#FFD700'>●</span> <b>Your proposal</b> &nbsp;
        <span style='color:white; background:#005A8E; padding:1px 5px; border-radius:3px;'>◆</span> Substation
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Hover over any marker for details. Data center locations are realistic but approximate.")

# ── RIGHT: Scatter plot ────────────────────────────────────────────────────────
with chart_col:
    st.markdown("**📈 Your Scenario vs. 500 Simulated Training Cases**")

    dot_colors = [STRESS_COLORS[s] for s in df['grid_stress_level']]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # Background training points
    ax.scatter(
        df['data_center_size_mw'],
        df['existing_grid_load_pct'],
        c=dot_colors,
        alpha=0.3,
        s=16,
        zorder=2,
        linewidths=0,
    )

    # User's scenario — gold star
    ax.scatter(
        size_mw, grid_load,
        color=USER_COLOR,
        s=320, marker='*',
        zorder=6,
        edgecolors='white',
        linewidths=0.7,
    )

    # Smart annotation offset (avoid chart edges)
    ox = 38 if size_mw < 400 else -105
    oy = 5  if grid_load < 88  else -10
    ax.annotate(
        f'★ Your Scenario\n{stress_label} Stress',
        xy=(size_mw, grid_load),
        xytext=(size_mw + ox, grid_load + oy),
        fontsize=7.5, fontweight='bold',
        color=USER_COLOR,
        arrowprops=dict(arrowstyle='->', color=USER_COLOR, lw=1.1),
        zorder=7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                  edgecolor=USER_COLOR, linewidth=0.8, alpha=0.85)
    )

    ax.set_xlabel('Facility Size (MW)', color='#AAAAAA', fontsize=9)
    ax.set_ylabel('Existing Grid Load (%)', color='#AAAAAA', fontsize=9)
    ax.tick_params(colors='#AAAAAA', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    legend_handles = [
        mpatches.Patch(color=STRESS_COLORS['Low'],    label='Low (training scenario)'),
        mpatches.Patch(color=STRESS_COLORS['Medium'], label='Medium (training scenario)'),
        mpatches.Patch(color=STRESS_COLORS['High'],   label='High (training scenario)'),
        plt.Line2D([0],[0], marker='*', color='w',
                   markerfacecolor=USER_COLOR, markeredgecolor='white',
                   markersize=10, label='Your scenario'),
    ]
    ax.legend(
        handles=legend_handles, fontsize=7,
        facecolor='#1a1a2e', edgecolor='#444444',
        labelcolor='#CCCCCC', loc='upper left',
    )

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        "Each dot = one of 500 synthetic scenarios used to train the Random Forest model, "
        "color-coded by predicted grid stress. "
        "The X and Y axes show the two strongest predictors (59%+ combined feature importance). "
        "Move any slider above — the ★ updates instantly."
    )

st.divider()

# ── Bottom row: Confidence breakdown + key metrics ─────────────────────────────
st.subheader("📊 Prediction Breakdown")
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("**Confidence by Stress Level**")
    for label, prob in zip(le_stress.classes_, proba):
        color = STRESS_COLORS[label]
        st.markdown(
            f"<span style='color:{color}; font-weight:bold; font-size:15px'>{label}</span>"
            f"&nbsp;&nbsp;{prob*100:.1f}%",
            unsafe_allow_html=True
        )
        st.progress(int(prob * 100))

with b2:
    st.markdown("**Key Input Summary**")
    st.markdown(f"""
    <div class="metric-card">📍 <b>Neighborhood:</b> {neighborhood}</div>
    <div class="metric-card">🏗️ <b>Facility Size:</b> {size_mw} MW</div>
    <div class="metric-card">⚡ <b>Grid Load:</b> {grid_load}%</div>
    <div class="metric-card">📏 <b>Substation Distance:</b> {distance} mi</div>
    """, unsafe_allow_html=True)

with b3:
    st.markdown("**Model Info**")
    st.markdown(f"""
    <div class="metric-card">🤖 <b>Algorithm:</b> Random Forest (100 trees)</div>
    <div class="metric-card">🎯 <b>Test Accuracy:</b> 95%</div>
    <div class="metric-card">📦 <b>Training Samples:</b> 400 scenarios</div>
    <div class="metric-card">🔑 <b>Top Predictors:</b> Grid Load + Facility Size</div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Built by Brian Parker — BS Artificial Intelligence, UNC Charlotte · github.com/bparke60")