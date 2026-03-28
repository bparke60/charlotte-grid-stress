import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

random.seed(42)
np.random.seed(42)

@st.cache_resource
def train_model():
    n = 500
    neighborhoods = ['University City', 'South End', 'NoDa', 'Steele Creek', 'Ballantyne',
                     'East Charlotte', 'Mooresville', 'Matthews', 'Huntersville', 'Concord']
    data = {
        'neighborhood': [random.choice(neighborhoods) for _ in range(n)],
        'data_center_size_mw': np.round(np.random.uniform(10, 500, n), 1),
        'existing_grid_load_pct': np.round(np.random.uniform(40, 95, n), 1),
        'distance_to_substation_miles': np.round(np.random.uniform(0.5, 15, n), 2),
        'residential_density': np.round(np.random.uniform(1, 10, n), 1),
        'num_existing_data_centers': np.random.randint(0, 8, n),
    }

    def classify_stress(row):
        score = 0
        if row['data_center_size_mw'] > 300: score += 3
        elif row['data_center_size_mw'] > 150: score += 2
        else: score += 1
        if row['existing_grid_load_pct'] > 80: score += 3
        elif row['existing_grid_load_pct'] > 65: score += 2
        else: score += 1
        if row['distance_to_substation_miles'] > 10: score += 2
        else: score += 1
        if row['num_existing_data_centers'] > 4: score += 2
        else: score += 1
        if score >= 8: return 'High'
        elif score >= 5: return 'Medium'
        else: return 'Low'

    df = pd.DataFrame(data)
    df['grid_stress_level'] = df.apply(classify_stress, axis=1)

    le_neighborhood = LabelEncoder()
    le_stress = LabelEncoder()
    df['neighborhood_encoded'] = le_neighborhood.fit_transform(df['neighborhood'])
    df['stress_encoded'] = le_stress.fit_transform(df['grid_stress_level'])

    features = ['neighborhood_encoded', 'data_center_size_mw', 'existing_grid_load_pct',
                'distance_to_substation_miles', 'residential_density', 'num_existing_data_centers']

    X = df[features]
    y = df['stress_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le_neighborhood, le_stress, features

model, le_neighborhood, le_stress, features = train_model()

st.title("Charlotte Data Center Grid Stress Predictor")
st.markdown("### Will a proposed data center stress Charlotte's power grid?")
st.markdown("This tool uses a machine learning model to predict grid stress level based on proposed data center characteristics. Dataset is synthetically generated using real-world grid engineering parameters.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    neighborhood = st.selectbox("Neighborhood", sorted([
        'University City', 'South End', 'NoDa', 'Steele Creek', 'Ballantyne',
        'East Charlotte', 'Mooresville', 'Matthews', 'Huntersville', 'Concord'
    ]))
    size_mw = st.slider("Data Center Size (MW)", min_value=10, max_value=500, value=150, step=10)
    grid_load = st.slider("Existing Grid Load (%)", min_value=40, max_value=95, value=65, step=1)

with col2:
    distance = st.slider("Distance to Nearest Substation (miles)", min_value=0.5, max_value=15.0, value=5.0, step=0.5)
    density = st.slider("Residential Density (1-10)", min_value=1, max_value=10, value=5, step=1)
    existing = st.slider("Existing Data Centers Nearby", min_value=0, max_value=8, value=2, step=1)

st.divider()

if st.button("Predict Grid Stress", type="primary"):
    hood_encoded = le_neighborhood.transform([neighborhood])[0]
    input_data = pd.DataFrame([[hood_encoded, size_mw, grid_load, distance, density, existing]],
                               columns=features)
    prediction = model.predict(input_data)[0]
    stress_label = le_stress.inverse_transform([prediction])[0]
    proba = model.predict_proba(input_data)[0]
    confidence = max(proba) * 100

    st.markdown("## Result")

    if stress_label == "High":
        st.error(f"HIGH Grid Stress — {confidence:.1f}% confidence")
        st.markdown("This proposal would likely place significant strain on Charlotte's power grid and may require major infrastructure upgrades before approval.")
    elif stress_label == "Medium":
        st.warning(f"MEDIUM Grid Stress — {confidence:.1f}% confidence")
        st.markdown("This proposal would place moderate strain on the grid. Further engineering review is recommended.")
    else:
        st.success(f"LOW Grid Stress — {confidence:.1f}% confidence")
        st.markdown("This proposal appears manageable for the existing grid infrastructure in this area.")

    st.markdown("### Confidence Breakdown")
    for label, prob in zip(le_stress.classes_, proba):
        st.progress(int(prob * 100), text=f"{label}: {prob*100:.1f}%")