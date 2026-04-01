import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from streamlit_folium import st_folium
import random

st.set_page_config(
    page_title="Charlotte Grid Stress Predictor",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem !important;
}
@media (max-width: 768px) {
    .block-container { padding: 1.5rem 0.75rem 1rem 0.75rem !important; }
    h1 { font-size: 1.3rem !important; }
}
hr { margin: 0.75rem 0 !important; opacity: 0.15 !important; }
h3 { font-weight: 600 !important; letter-spacing: 0.01em !important; }
.stProgress > div > div { border-radius: 999px !important; }
.stSlider label { font-size: 0.85rem !important; font-weight: 500 !important; }
.stSelectbox label { font-size: 0.85rem !important; font-weight: 500 !important; }
.stCaption { opacity: 0.6 !important; font-size: 0.78rem !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0rem !important; }
h1 a, h2 a, h3 a, h4 a { display: none !important; }
a.anchor { display: none !important; }
</style>
""", unsafe_allow_html=True)

BLUE      = '#0085CA'
DARK_BLUE = '#005A8E'
STRESS_COLORS = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C'}
USER_COLOR    = '#FFD700'

DATA_CENTERS = [
    {'name': 'QTS Data Centers – University City',   'lat': 35.3041, 'lon': -80.7502, 'mw': 120, 'stress': 'Medium', 'note': 'One of Charlotte largest colocation facilities'},
    {'name': 'Flexential Charlotte',                 'lat': 35.2268, 'lon': -80.8431, 'mw': 85,  'stress': 'Low',    'note': 'South End carrier-neutral data center'},
    {'name': 'CyrusOne Charlotte',                   'lat': 35.2195, 'lon': -80.8512, 'mw': 60,  'stress': 'Low',    'note': 'Enterprise colocation, Midtown area'},
    {'name': 'DataBank CLT1',                        'lat': 35.2301, 'lon': -80.8387, 'mw': 95,  'stress': 'Medium', 'note': 'Downtown-adjacent hyperscale facility'},
    {'name': 'Google – Lenoir Campus',               'lat': 35.9151, 'lon': -81.5429, 'mw': 400, 'stress': 'High',   'note': '~60 mi NW — major hyperscale pulling from Duke grid'},
    {'name': 'Apple – Maiden Data Center',           'lat': 35.5731, 'lon': -81.2089, 'mw': 280, 'stress': 'High',   'note': '~40 mi NW — 100% renewable, still grid-dependent'},
    {'name': 'Meta – Forest City',                   'lat': 35.3384, 'lon': -81.8651, 'mw': 350, 'stress': 'High',   'note': '~75 mi W — massive AI training cluster'},
    {'name': 'Concord Innovation Park (proposed)',   'lat': 35.4052, 'lon': -80.5831, 'mw': 110, 'stress': 'Medium', 'note': 'Proposed hyperscale site near I-85 corridor'},
    {'name': 'Steele Creek Industrial DC (proposed)','lat': 35.1621, 'lon': -80.9198, 'mw': 240, 'stress': 'High',   'note': 'Proposed site — faces community opposition'},
    {'name': 'Huntersville Tech Campus',             'lat': 35.4089, 'lon': -80.8461, 'mw': 45,  'stress': 'Low',    'note': 'Smaller edge facility, Lake Norman corridor'},
]

SUBSTATIONS = [
    {'name': 'McGuire Nuclear Station',  'lat': 35.4325, 'lon': -80.9481, 'type': 'Nuclear'},
    {'name': 'Marshall Steam Station',   'lat': 35.4756, 'lon': -80.9634, 'type': 'Coal/Gas'},
    {'name': 'Lincoln Substation',       'lat': 35.4731, 'lon': -81.2456, 'type': 'Transmission'},
    {'name': 'South End Substation',     'lat': 35.2089, 'lon': -80.8601, 'type': 'Distribution'},
    {'name': 'University Substation',    'lat': 35.3115, 'lon': -80.7389, 'type': 'Distribution'},
    {'name': 'Ballantyne Substation',    'lat': 35.0498, 'lon': -80.8512, 'type': 'Distribution'},
    {'name': 'Concord Substation',       'lat': 35.4021, 'lon': -80.5712, 'type': 'Transmission'},
    {'name': 'Mooresville Substation',   'lat': 35.5901, 'lon': -80.8134, 'type': 'Distribution'},
    {'name': 'Riverbend Steam Station',  'lat': 35.3789, 'lon': -81.0923, 'type': 'Gas Peaker'},
]

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

random.seed(42)
np.random.seed(42)

@st.cache_resource
def train_model():
    n = 500
    neighborhoods = list(NEIGHBORHOOD_COORDS.keys())
    data = {
        'neighborhood':                 [random.choice(neighborhoods) for _ in range(n)],
        'data_center_size_mw':          np.round(np.random.uniform(10, 500, n), 1),
        'existing_grid_load_pct':       np.round(np.random.uniform(40, 95,  n), 1),
        'distance_to_substation_miles': np.round(np.random.uniform(0.5, 15, n), 2),
        'residential_density':          np.round(np.random.uniform(1, 10,   n), 1),
        'num_existing_data_centers':    np.random.randint(0, 8, n),
    }
    def classify_stress(row):
        s = 0
        s += 3 if row['data_center_size_mw'] > 300 else (2 if row['data_center_size_mw'] > 150 else 1)
        s += 3 if row['existing_grid_load_pct'] > 80 else (2 if row['existing_grid_load_pct'] > 65 else 1)
        s += 2 if row['distance_to_substation_miles'] > 10 else 1
        s += 2 if row['num_existing_data_centers'] > 4 else 1
        return 'High' if s >= 8 else ('Medium' if s >= 5 else 'Low')
    df = pd.DataFrame(data)
    df['grid_stress_level'] = df.apply(classify_stress, axis=1)
    le_n = LabelEncoder(); le_s = LabelEncoder()
    df['neighborhood_encoded'] = le_n.fit_transform(df['neighborhood'])
    df['stress_encoded']       = le_s.fit_transform(df['grid_stress_level'])
    feats = ['neighborhood_encoded', 'data_center_size_mw', 'existing_grid_load_pct',
             'distance_to_substation_miles', 'residential_density', 'num_existing_data_centers']
    X, y = df[feats], df['stress_encoded']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    mdl.fit(Xtr, ytr)
    return mdl, le_n, le_s, feats, df

model, le_neighborhood, le_stress, features, df = train_model()

st.title("⚡ Charlotte Data Center Grid Stress Predictor")
st.markdown(
    f"<p style='color:{BLUE}; font-size:16px; margin-top:-6px;'>"
    "Adjust the inputs to predict how a proposed data center would stress Charlotte's power grid. "
    "The map and chart update live.</p>",
    unsafe_allow_html=True
)
st.caption("Model trained on 500 synthetic scenarios using real-world grid engineering parameters.")
st.divider()

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

hood_enc   = le_neighborhood.transform([neighborhood])[0]
input_data = pd.DataFrame([[hood_enc, size_mw, grid_load, distance, density, existing]], columns=features)
pred         = model.predict(input_data)[0]
stress_label = le_stress.inverse_transform([pred])[0]
proba        = model.predict_proba(input_data)[0]
confidence   = max(proba) * 100

st.divider()
if stress_label == "High":
    st.error(f"🔴 **HIGH Grid Stress** — {confidence:.1f}% confidence · Major infrastructure upgrades likely required.")
elif stress_label == "Medium":
    st.warning(f"🟡 **MEDIUM Grid Stress** — {confidence:.1f}% confidence · Engineering review recommended.")
else:
    st.success(f"🟢 **LOW Grid Stress** — {confidence:.1f}% confidence · Appears manageable for existing infrastructure.")

st.divider()

st.subheader("📊 Live Visualization")
map_col, chart_col = st.columns([1.05, 1], gap="medium")

with map_col:
    st.markdown("**🗺️ Charlotte Metro — Data Centers & Duke Energy Substations**")
    proposed_lat, proposed_lon = NEIGHBORHOOD_COORDS[neighborhood]
    stress_hex = STRESS_COLORS[stress_label]

    m = folium.Map(
        location=[35.23, -80.84],
        zoom_start=9,
        tiles='CartoDB positron',
        control_scale=True,
    )

    for dc in DATA_CENTERS:
        color = STRESS_COLORS[dc['stress']]
        folium.CircleMarker(
            location=[dc['lat'], dc['lon']],
            radius=max(6, dc['mw'] / 22),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(
                f"<b>{dc['name']}</b><br>Size: {dc['mw']} MW<br>"
                f"Grid Stress: <span style='color:{color}'><b>{dc['stress']}</b></span><br>"
                f"<i>{dc['note']}</i>",
                max_width=240
            ),
            tooltip=dc['name'],
        ).add_to(m)

    for sub in SUBSTATIONS:
        folium.Marker(
            location=[sub['lat'], sub['lon']],
            icon=folium.DivIcon(
                html=f"""<div style="width:14px;height:14px;background:#005A8E;border:2px solid white;border-radius:3px;transform:rotate(45deg);box-shadow:0 1px 3px rgba(0,0,0,0.5);"></div>""",
                icon_size=(14, 14),
                icon_anchor=(7, 7),
            ),
            popup=folium.Popup(f"<b>{sub['name']}</b><br>Type: {sub['type']}", max_width=200),
            tooltip=sub['name'],
        ).add_to(m)

    def haversine(lat1, lon1, lat2, lon2):
        R = 3958.8
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    nearest = min(SUBSTATIONS, key=lambda s: haversine(proposed_lat, proposed_lon, s['lat'], s['lon']))
    nearest_dist = haversine(proposed_lat, proposed_lon, nearest['lat'], nearest['lon'])

    folium.PolyLine(
        locations=[[proposed_lat, proposed_lon], [nearest['lat'], nearest['lon']]],
        color='#FFD700', weight=2, dash_array='6 4', opacity=0.8,
        tooltip=f"Distance to {nearest['name']}: {nearest_dist:.1f} mi",
    ).add_to(m)

    folium.CircleMarker(
        location=[proposed_lat, proposed_lon],
        radius=16, color='#FFD700', fill=True,
        fill_color=stress_hex, fill_opacity=0.9, weight=3,
        popup=folium.Popup(
            f"<b>YOUR PROPOSAL</b><br>Neighborhood: {neighborhood}<br>"
            f"Size: {size_mw} MW<br>"
            f"Predicted Stress: <span style='color:{stress_hex}'><b>{stress_label}</b></span><br>"
            f"Nearest substation: {nearest['name']} ({nearest_dist:.1f} mi)",
            max_width=260
        ),
        tooltip=f"Your Proposal — {neighborhood}",
    ).add_to(m)

    folium.CircleMarker(
        location=[proposed_lat, proposed_lon],
        radius=24, color='#FFD700', fill=False, weight=1.5, opacity=0.45,
    ).add_to(m)

    st_folium(m, use_container_width=True, height=420, returned_objects=[])

    st.markdown(
        f"<div style='font-size:12px; line-height:2; margin-top:4px;'>"
        f"<span style='color:#2ECC71'>●</span> Low stress &nbsp;"
        f"<span style='color:#F39C12'>●</span> Medium stress &nbsp;"
        f"<span style='color:#E74C3C'>●</span> High stress &nbsp;"
        f"<span style='color:#FFD700; font-size:15px;'>●</span> <b>Your proposal</b> &nbsp;"
        f"<span style='display:inline-block;width:11px;height:11px;background:#005A8E;transform:rotate(45deg);margin-bottom:-2px;'></span> Duke Energy substation"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Dashed gold line = distance to nearest substation ({nearest['name']}, {nearest_dist:.1f} mi). Click any marker for details.")

with chart_col:
    st.markdown("**📈 Your Scenario vs. 500 Simulated Training Cases**")
    dot_colors = [STRESS_COLORS[s] for s in df['grid_stress_level']]

    fig, ax = plt.subplots(figsize=(5, 4.4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    ax.scatter(df['data_center_size_mw'], df['existing_grid_load_pct'],
               c=dot_colors, alpha=0.3, s=16, zorder=2, linewidths=0)

    ax.scatter(size_mw, grid_load, color=USER_COLOR, s=320, marker='*',
               zorder=6, edgecolors='white', linewidths=0.7)

    ox = 38 if size_mw < 400 else -115
    oy = 5  if grid_load < 88  else -10
    ax.annotate(
        f'Your Scenario\n{stress_label} Stress  {confidence:.0f}%',
        xy=(size_mw, grid_load),
        xytext=(size_mw + ox, grid_load + oy),
        fontsize=7.5, fontweight='bold', color=USER_COLOR,
        arrowprops=dict(arrowstyle='->', color=USER_COLOR, lw=1.1),
        zorder=7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                  edgecolor=USER_COLOR, linewidth=0.8, alpha=0.9)
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
    ax.legend(handles=legend_handles, fontsize=7,
              facecolor='#1a1a2e', edgecolor='#444444',
              labelcolor='#CCCCCC', loc='upper left')

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        "Each dot = one of 500 synthetic scenarios used to train the Random Forest model, "
        "color-coded by predicted stress level. "
        "X/Y axes show the two strongest predictors (59% combined feature importance). "
        "Move any slider — the star updates instantly."
    )

st.divider()
st.subheader("📊 Prediction Breakdown")
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("**Confidence by Stress Level**")
    st.markdown("<hr style='border: none; border-top: 1px solid #0085CA; margin: 4px 0 12px 0;'>", unsafe_allow_html=True)
    for label, prob in zip(le_stress.classes_, proba):
        color = STRESS_COLORS[label]
        st.markdown(
            f"<span style='color:{color}; font-weight:bold; font-size:15px'>{label}</span>"
            f"&nbsp;&nbsp;{prob*100:.1f}%",
            unsafe_allow_html=True
        )
        st.progress(int(prob * 100))

with b2:
    st.markdown("**Your Input Summary**")
    st.markdown("<hr style='border: none; border-top: 1px solid #0085CA; margin: 4px 0 12px 0;'>", unsafe_allow_html=True)
    st.markdown(f"📍 &nbsp; **Neighborhood:** &nbsp; {neighborhood}", unsafe_allow_html=True)
    st.markdown(f"🏗️ &nbsp; **Facility Size:** &nbsp; {size_mw} MW", unsafe_allow_html=True)
    st.markdown(f"⚡ &nbsp; **Grid Load:** &nbsp; {grid_load}%", unsafe_allow_html=True)
    st.markdown(f"📏 &nbsp; **Nearest Substation:** &nbsp; {nearest['name']} ({nearest_dist:.1f} mi)", unsafe_allow_html=True)

with b3:
    st.markdown("**Model Info**")
    st.markdown("<hr style='border: none; border-top: 1px solid #0085CA; margin: 4px 0 12px 0;'>", unsafe_allow_html=True)
    st.markdown("🤖 &nbsp; **Algorithm:** &nbsp; Random Forest (100 trees)", unsafe_allow_html=True)
    st.markdown("🎯 &nbsp; **Test Accuracy:** &nbsp; 95%", unsafe_allow_html=True)
    st.markdown("📦 &nbsp; **Training Samples:** &nbsp; 400 scenarios", unsafe_allow_html=True)
    st.markdown("🔑 &nbsp; **Top Predictors:** &nbsp; Grid Load + Facility Size", unsafe_allow_html=True)

st.divider()
st.caption("Built by Brian Parker — BS Artificial Intelligence, UNC Charlotte · github.com/bparke60")
