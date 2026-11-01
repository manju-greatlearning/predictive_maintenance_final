# Predictive Maintenance - Capstone 3 (Final)

End-to-end MLOps project for predicting machine failure using the AI4I 2020
Predictive Maintenance dataset from the UCI Machine Learning Repository.

## Repository layout
- `data/` - registered dataset, cleaned dataset, train/test splits
- `src/validate_dataset.py` - data registration and validation summary
- `src/prepare_data.py` - cleaning and stratified train/test split
- `src/train.py` - model tuning, metric logging and best-model persistence
- `app.py` - Streamlit deployment app that loads `models/best_model.joblib`
- `.github/workflows/pipeline.yml` - automated validate -> prepare -> artifact -> train -> app smoke-test -> commit workflow
- `notebooks/` - executed notebook used as report evidence
- `submission/` - final PDF report and executed HTML notebook for upload

## Run locally
```bash
pip install -r requirements.txt
python src/validate_dataset.py
python src/prepare_data.py
python src/train.py
streamlit run app.py
```

On macOS, XGBoost may require OpenMP (`brew install libomp`). The GitHub Actions
runner uses Linux.

## Deployment
Deploy on Streamlit Community Cloud with:
- Repository: this GitHub repository
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`

Dataset source: AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository,
https://doi.org/10.24432/C5HS5C.
