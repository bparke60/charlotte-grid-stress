# Charlotte Data Center Grid Stress Predictor

**Interactive machine-learning demonstration** exploring how hypothetical data-center characteristics may relate to modeled grid-stress categories around Charlotte.

**Live demo:** https://charlotte-grid-stress.streamlit.app

## Why this project matters
Growing demand from AI infrastructure raises practical questions about electrical capacity, site selection, and data-center planning. I built this project to explore those questions through Python, data visualization, and a classifier.

## Features
The Streamlit app lets users vary facility size, existing grid load, substation distance, residential density, nearby data centers, and location. It displays a modeled Low/Medium/High stress category, a confidence breakdown, scenario visualization, and a map.

## Stack
Python, pandas, NumPy, scikit-learn, Streamlit, matplotlib, folium, and streamlit-folium.

## Model and reported evaluation
- Random Forest classifier, 100 estimators
- 500 **synthetically generated** scenarios
- Reported split: 400 training scenarios / 100 test scenarios
- Reported held-out accuracy: **95% on synthetic scenarios**

**Interpretation:** This accuracy measures how well the model predicts labels within the generated scenario dataset. It does **not** establish 95% accuracy for real Charlotte grid operations or validate a real-world engineering risk assessment. Confidence scores are model outputs, not calibrated probabilities unless calibration is separately demonstrated.

## Dataset limitations
The repository does not use proprietary Duke Energy operational measurements. Inputs and labels are generated for educational exploration, so the classifier may learn patterns imposed by the simulation rules. The app should not be used for infrastructure siting, utility planning, investment, or safety decisions.

## Run locally
```bash
git clone https://github.com/bparke60/charlotte-grid-stress.git
cd charlotte-grid-stress
python -m pip install -r requirements.txt
streamlit run app.py
```

## Repository
- `app.py` — interactive app and model workflow
- `charlotte_grid_stress.ipynb` — exploration and model development
- `requirements.txt` — dependencies
- `portfolio/` — independent Jupyter coursework-inspired data engineering extensions

## Next engineering steps
1. Separate data generation, training, and inference into testable modules.
2. Add deterministic seeds, input-validation tests, and a model evaluation report.
3. Compare the random forest to a simple baseline and evaluate per-class precision/recall.
4. Introduce public measured data if available, with documented provenance.

**Author:** Brian Parker | UNC Charlotte B.S. Artificial Intelligence
