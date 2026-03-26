# Jordan Sneaker Profitability Prediction (SVM + Streamlit)

Predict whether a Jordan sneaker transaction is likely to be profitable using a tuned Support Vector Machine (SVM) model, then serve predictions through a Streamlit web app.

## Overview

This project uses a tabular resale dataset and builds a binary classifier:

- Target: `Target_Profitable` (1 if `Profit_Margin_USD > 0`, else 0)
- Model: `SVC` with preprocessing pipeline (`OneHotEncoder` + `StandardScaler`)
- UI: Streamlit app for interactive inference

The app predicts profitability from user inputs such as shoe model, colorway, condition, sales channel, retail price, and inventory days.

## Repository Structure

```text
.
├── app.py
├── model.ipynb
├── jordan_market_dataset_2026.csv
└── artifacts
    ├── svm_profitability_pipeline.joblib
    └── model_metadata.json
```

## Dataset

Input dataset includes columns like:

- `Transaction_ID`
- `Sale_Date`
- `Shoe_Model`
- `Colorway`
- `Condition`
- `Retail_Price_USD`
- `Resale_Price_USD`
- `Sales_Channel`
- `Days_in_Inventory`
- `Profit_Margin_USD`

## Problem Definition

### Objective

Classify each transaction as:

- Profitable (`1`)
- Not profitable (`0`)

### Leakage Prevention

To avoid data leakage, these columns are removed from training features:

- `Transaction_ID` (identifier)
- `Sale_Date` (time identifier in current setup)
- `Resale_Price_USD` (unknown at prediction time)
- `Profit_Margin_USD` (directly defines the target)

Final training features:

- Categorical: `Shoe_Model`, `Colorway`, `Condition`, `Sales_Channel`
- Numeric: `Retail_Price_USD`, `Days_in_Inventory`

## Modeling Pipeline

### Preprocessing

- `OneHotEncoder(handle_unknown="ignore")` for categorical columns
- `StandardScaler` for numeric columns
- Combined using `ColumnTransformer`

### Model

- Baseline: `SVC(kernel="rbf", probability=True)`
- Tuning: `GridSearchCV` over:
  - `kernel`: `rbf`, `linear`
  - `C`: `0.1, 1, 10, 50`
  - `gamma`: `scale, 0.01, 0.1`
  - `class_weight`: `None, balanced`

### Best Parameters

```python
{
  "model__C": 10,
  "model__class_weight": "balanced",
  "model__gamma": 0.1,
  "model__kernel": "rbf"
}
```

### Observed Performance

- Best CV ROC-AUC: `0.9671`
- Test Accuracy: `0.9430`
- Test ROC-AUC: `0.9667`
- Confusion Matrix:

```text
[[333  54]
 [  3 610]]
```

## Local Setup

## 1) Create and activate virtual environment (recommended)

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install --upgrade pip
pip install streamlit pandas scikit-learn joblib numpy
```

## 3) Run the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Reproducing Training (Notebook Workflow)

Open `model.ipynb` and run cells in order:

1. Load libraries and dataset
2. EDA and sanity checks
3. Build target and leakage-safe features
4. Train/test split + preprocessing
5. Baseline SVM training and evaluation
6. GridSearchCV tuning
7. Save artifacts to `artifacts/`

Saved outputs:

- `artifacts/svm_profitability_pipeline.joblib`
- `artifacts/model_metadata.json`

## Streamlit App Behavior

The app shows:

- Probability of profitability (`predict_proba`)
- Model default decision (`model.predict`)
- Threshold-based decision (user-controlled threshold slider)

This helps compare pure model output vs custom risk threshold.

## Deployment (Streamlit Community Cloud)

1. Push project to a GitHub repository
2. Go to Streamlit Community Cloud and create a new app
3. Connect the repository
4. Set:
   - Main file path: `app.py`
5. Ensure repository includes:
   - `app.py`
   - `artifacts/` folder
   - `jordan_market_dataset_2026.csv`
   - dependency list (recommended: `requirements.txt`)

Recommended `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

## Troubleshooting

### 1) Model predicts unexpectedly after environment changes

If you see warnings about scikit-learn version mismatch while loading `.joblib`, retrain and resave artifacts in the current environment, then restart Streamlit.

### 2) App always predicts one class

- Check threshold slider (try `0.45` to `0.55`)
- Validate app displays both model default and threshold-based decisions
- Test with realistic combinations from your dataset

### 3) Missing artifact files

Re-run the artifact-saving cell in notebook and confirm files exist in `artifacts/`.

## Future Improvements

- Add probability calibration (`CalibratedClassifierCV`)
- Add model explainability (SHAP or permutation importance)
- Save a separate training script (`train.py`) for one-command retraining
- Add automated tests for inference schema and prediction outputs
- Add CI workflow for linting and app startup checks

## License

Add your preferred license (for example, MIT) before public release.
