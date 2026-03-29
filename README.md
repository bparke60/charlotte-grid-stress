# Charlotte Data Center Grid Stress Predictor

A machine learning web app that predicts how much strain a proposed data center would place on Charlotte's power grid — built in response to the real and ongoing debate over AI infrastructure expansion in the Charlotte metro area.

🔗 **Live App:** https://charlotte-grid-stress.streamlit.app

## Why I Built This

Charlotte is currently at the center of a national conversation about AI data centers. Duke Energy has projected that electricity demand will grow 8x faster over the next 15 years, largely driven by AI infrastructure. Neighborhoods like East Charlotte and Matthews have already seen proposed data centers face community opposition over grid capacity concerns.

I wanted to see if machine learning could help quantify that impact before it becomes a town hall battle.

## What It Does

Users input the characteristics of a proposed data center and the app predicts grid stress level in real time — no button needed, results update as you adjust the inputs:

- Neighborhood location
- Facility size (MW)
- Existing grid load in the area (%)
- Distance to nearest substation (miles)
- Residential density
- Number of existing data centers nearby

The model returns a predicted grid stress level — **Low**, **Medium**, or **High** — along with a live confidence breakdown and a dynamic scatter plot showing how your scenario compares to all 500 training scenarios.

## Model Performance

- **Algorithm:** Random Forest Classifier (100 estimators)
- **Accuracy:** 95% on held-out test data
- **Training samples:** 400
- **Test samples:** 100

### Key Findings

The model identified existing grid load and data center size as the two strongest predictors of grid stress — accounting for over 59% of feature importance combined. Neighborhood location alone was the weakest predictor, suggesting no area is inherently safe if the grid is already strained.

## Tech Stack

- **Python** — core language
- **pandas** — data manipulation
- **scikit-learn** — Random Forest model, train/test split, evaluation
- **NumPy** — numerical operations
- **Streamlit** — interactive web app interface
- **matplotlib** — dynamic scatter plot visualization

## Dataset

The dataset is synthetically generated using real-world grid engineering parameters. Duke Energy's operational grid data is not publicly available, so 500 realistic scenarios were simulated using ranges consistent with industry standards. This is a common approach in ML projects where operational data is proprietary.

## How To Run
```bash
git clone https://github.com/bparke60/charlotte-grid-stress.git
cd charlotte-grid-stress
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
charlotte-grid-stress/
├── app.py                       # Streamlit web app + ML model
├── charlotte_grid_stress.ipynb  # Model development notebook
├── requirements.txt             # Dependencies
└── README.md
```

## Author

Brian Parker — BS Artificial Intelligence, UNC Charlotte  
[github.com/bparke60](https://github.com/bparke60)- folium + streamlit-folium — interactive map visualization
