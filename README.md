# Charlotte Data Center Grid Stress Predictor

A machine learning web app that predicts how much strain a proposed data center would place on Charlotte's power grid — built in response to the real and ongoing debate over AI infrastructure expansion in the Charlotte metro area.

🔗 **Live App:** https://charlotte-grid-stress.streamlit.app

---

## Why I Built This

Charlotte is at the center of a national conversation about AI data centers. Duke Energy has projected electricity demand will grow 8x faster over the next 15 years, largely driven by AI infrastructure. Neighborhoods like East Charlotte and Matthews have already seen proposed data centers face community opposition over grid capacity concerns.

I wanted to see if machine learning could help quantify that impact before it becomes a town hall battle.

---

## What It Does

Input the characteristics of a proposed data center and the app predicts grid stress level in real time — results update instantly as you adjust the sliders:

- 📍 Neighborhood location
- 🏗️ Facility size (MW)
- ⚡ Existing grid load in the area (%)
- 📏 Distance to nearest substation (miles)
- 🏘️ Residential density
- 🖥️ Number of existing data centers nearby

The model returns a predicted stress level — **Low**, **Medium**, or **High** — along with a live confidence breakdown, an interactive map of Charlotte's data center landscape, and a dynamic scatter plot showing how your scenario compares to all 500 training scenarios.

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier (100 estimators) |
| Test Accuracy | 95% |
| Training Samples | 400 scenarios |
| Test Samples | 100 scenarios |
| Top Predictors | Grid Load + Facility Size (59% combined feature importance) |

---

## Key Findings

Existing grid load and data center size are the two strongest predictors of grid stress, accounting for over 59% of feature importance combined. Neighborhood location alone is the weakest predictor — suggesting no area is inherently safe if the grid is already strained.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | Random Forest model, train/test split, evaluation |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| Streamlit | Interactive web app interface |
| matplotlib | Dynamic scatter plot visualization |
| folium + streamlit-folium | Interactive map visualization |

---

## Dataset

The dataset is synthetically generated using real-world grid engineering parameters. Duke Energy's operational grid data is not publicly available, so 500 realistic scenarios were simulated using ranges consistent with industry standards. This is a common approach in ML projects where operational data is proprietary.

---

## How To Run
```bash
git clone https://github.com/bparke60/charlotte-grid-stress.git
cd charlotte-grid-stress
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure
```
charlotte-grid-stress/
├── app.py                        # Streamlit web app + ML model
├── charlotte_grid_stress.ipynb   # Model development notebook
├── requirements.txt              # Dependencies
└── README.md
```

---

## Author

**Brian Parker** — BS Artificial Intelligence, UNC Charlotte  
github.com/bparke60

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
© 2026 Brian Parker. You may not use this project or its contents for commercial purposes.
