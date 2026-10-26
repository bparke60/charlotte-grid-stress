# Jupyter Machine Learning & Data Engineering Portfolio

Independent portfolio extensions of my UNC Charlotte ITCS 3156 Jupyter coursework (2026).

## Projects
1. **Data Pipeline & Regression** — Python, Pandas, scikit-learn, SQLite; dataset validation, SQL round-trip, linear/ridge regression.
2. **Gradient Descent** — NumPy implementation of batch gradient descent and training loss tracking.
3. **Python Algorithms** — Python list and matrix transformations, histogram and recursion exercises.

The original homework motivated these extensions; the extensions are separate from course submissions. All datasets used are public and no private course test cases or instructor solutions are included.

## Run
```bash
python -m pip install -r portfolio/requirements.txt
python portfolio/pipeline.py
python portfolio/gradient_descent.py
python -m pytest portfolio/tests -q
```

The pipeline uses scikit-learn's public diabetes dataset (442 records), writes it to a local SQLite database, and compares two models using a reproducible train/test split. No clinical predictions should be made from this educational project.
