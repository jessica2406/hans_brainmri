# HAN-S: Subdural Empyema Detection
Highly-Adaptive Neuro-Symbolic System.

## Setup Instructions
1. Clone the repo.
2. Create a virtual environment: `python -m venv hans_env`
3. Activate it and run: `pip install -r requirements.txt`
4. Place your images in `data/empyema/` and `data/raw/`.
5. Run `python scripts/preprocess.py` to prepare data.
6. Run `streamlit run app.py` to start the UI.