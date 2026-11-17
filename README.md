# SmartShield — Credit Card Fraud Detection and Risk Assessment System

SmartShield is a machine-learning project that analyses credit-card transaction data, predicts whether a transaction is likely to be fraudulent, and assigns a risk score and risk category. The project includes a model-training workflow, saved model artifacts, evaluation results, and a Streamlit dashboard for batch CSV predictions.

> **Project status:** Academic/coursework prototype. Predictions are for learning and demonstration, not for making real financial or credit decisions.

## Key features

- Upload a CSV of transactions through the Streamlit dashboard.
- Accept the 30 transaction input features (`Time`, `V1`–`V28`, and `Amount`). An optional `Class` column in an uploaded CSV is ignored during prediction.
- Display the number of transactions processed and the counts predicted as fraudulent or legitimate.
- Show each transaction's fraud probability, risk score, risk level, and prediction.
- Compare candidate machine-learning models and keep model evaluation outputs in the repository.
- Include an experiment comparing model performance with and without SMOTE, as recorded in the results files.

## Repository structure

**GitHub repository:** <https://github.com/rahulkumawat835/Credit-Card-Fraud-Detection-Using-Machine-Learning>

The published repository is organised as follows:

```text
Credit-Card-Fraud-Detection-Using-Machine-Learning/
├── dashboard/
│   ├── models/
│   │   ├── smartshield_model.joblib
│   │   └── smartshield_scaler.joblib
│   ├── app.py
│   ├── config.json
│   ├── demo_transactions.csv
│   ├── final_model_results.csv
│   ├── requirements.txt
│   └── smote_comparison.csv
├── notebooks/
│   └── Credit Card Fraud Detection.py
├── results/
│   ├── Comparision Table.png
│   ├── F1 score.png
│   ├── PR-AUC.png
│   ├── Pricision.png
│   ├── Recall.png
│   └── SmartShield_Model_Comparison.csv
└── .gitignore
```

The `results/` folder contains the model-comparison CSV and the evaluation plots shown in the GitHub repository: the comparison table, F1-score, PR-AUC, precision, and recall images. The original dataset is not stored in this repository. Download it separately using the instructions below. The `reference/` folder used during local development is intentionally not part of this repository.

## Dataset

SmartShield uses the **Credit Card Fraud Detection** dataset published on Kaggle by the Machine Learning Group of ULB (MLG-ULB):

- Dataset page: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download>
- Download the file named `creditcard.csv`.
- Place it in a local `data/` directory at the project root when running the training workflow:

```text
project-root/
└── data/
    └── creditcard.csv
```

The dataset contains anonymised transaction features. `Class` is the target label (`0` = legitimate, `1` = fraudulent). The input features used for prediction are `Time`, `V1`–`V28`, and `Amount`.

**Dataset access and terms:** Download the dataset from Kaggle and follow the dataset page's access and usage terms. The dataset is excluded from this GitHub repository to avoid committing a large data file.

## Technology stack

- Python
- pandas and NumPy for data handling
- scikit-learn for preprocessing, model training, and evaluation
- XGBoost for gradient-boosted tree modelling
- joblib for saving and loading model artifacts
- Streamlit for the interactive dashboard
- SMOTE for the class-imbalance experiment, where included in the training workflow

## Run the Streamlit dashboard (Windows PowerShell)

Run these commands from the repository root.

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the dashboard dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\dashboard\requirements.txt
```

3. Launch the dashboard:

```powershell
python -m streamlit run .\dashboard\app.py
```

4. Open the local URL printed by Streamlit, usually <http://localhost:8501>.

### Input CSV format

The uploaded CSV must contain one column for each of the 30 model input features:

```text
Time, V1, V2, ..., V28, Amount
```

The actual CSV header should use the exact feature names (without the ellipsis). An optional `Class` column may be present, but it is not used by the dashboard to make predictions. The supplied `dashboard/demo_transactions.csv` can be used as a demonstration input file.

## Reproducing the training workflow

The training workflow is in `notebooks/Credit Card Fraud Detection.py`. To reproduce it:

1. Download `creditcard.csv` from the Kaggle dataset page and place it in `data/creditcard.csv`.
2. Open the training script and ensure its dataset path points to that file (or upload the CSV and adapt the path when using Google Colab).
3. Run the script from top to bottom.
4. Review the generated evaluation and comparison results. The repository includes the saved comparison outputs under `results/` and `dashboard/`.
5. Save or replace the trained model and scaler artifacts in `dashboard/models/` only after checking that the dashboard is compatible with the newly generated artifacts.

### Reproducibility notes

- The dataset is imbalanced; use stratified splitting so the fraud/legitimate class proportions are represented in both training and test sets.
- Fit preprocessing steps such as scaling on the training data only, then apply the fitted transformer to validation/test data. Do not fit a scaler on the full dataset before splitting.
- Apply SMOTE, if used, to the training data only. Do not resample the validation or test sets.
- Keep the random seeds, split settings, feature list, preprocessing steps, and selected model settings recorded in the training script or experiment notes. Use the same settings when comparing runs.
- The saved `.joblib` model and scaler should be loaded with compatible versions of their training libraries. scikit-learn does not guarantee cross-version compatibility for persisted estimators. Record the exact environment used for training and prediction.
- The dependency file lists package requirements, but unpinned or minimum-version dependencies can resolve to newer releases over time. For a more exact reproduction, save the resolved environment after installation:

```powershell
python --version
python -m pip freeze > dashboard/requirements-lock.txt
```

Commit `dashboard/requirements-lock.txt` if you want to preserve the exact package versions used for a particular working run. For a Colab training run, also record the runtime's Python and package versions (for example, with `!python --version` and `!pip freeze`).

## Model outputs and risk interpretation

The dashboard reports:

- **Fraud probability:** the model's estimated probability for the fraud class, displayed as a percentage.
- **Risk score:** the probability converted to a 0–100 scale.
- **Risk level:** a Low, Medium, or High category based on the thresholds configured by the project.
- **Prediction:** the model's predicted class (fraudulent or legitimate).

Risk categories are project-defined demonstration bands, not thresholds certified by a bank, card network, or regulator. A low-risk prediction does not guarantee that a transaction is legitimate, and a high-risk prediction does not prove fraud.

## Known limitations

- The dataset uses anonymised features (`V1`–`V28`) and does not provide identifiable cardholder or merchant histories for customer-level behavioural analysis.
- The model learns patterns in the supplied historical dataset. Real-world fraud patterns can change, and performance on a new population may differ.
- The dashboard's predictions are only as reliable as the model, preprocessing pipeline, input data, and chosen decision threshold.
- This project is an academic prototype and has not been validated for production payment processing.

## AI-use disclosure

OpenAI ChatGPT was used during development as an assistance tool for project planning, explanations of machine-learning and Python workflows, troubleshooting environment/Git/Streamlit issues, and drafting project documentation. The project owner ran the code and commands, reviewed the outputs, and is responsible for checking the implementation, experiment results, and final submission. Any AI-suggested code or text should be reviewed and adapted by the project owner before use. No claim is made that AI independently conducted or validated the experiments.

## References and acknowledgements

1. **Dataset:** Machine Learning Group — ULB (MLG-ULB), *Credit Card Fraud Detection*, Kaggle. <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download>
2. **Related project consulted as a learning reference:** <https://github.com/hsamier/credit-card-fraud-detection>. This repository is acknowledged as a reference consulted during development; SmartShield is maintained in its own repository.
3. **Streamlit documentation:** <https://docs.streamlit.io/>
4. **scikit-learn documentation:** <https://scikit-learn.org/stable/>
5. **XGBoost documentation:** <https://xgboost.readthedocs.io/>
6. **pandas documentation:** <https://pandas.pydata.org/docs/>
7. **NumPy documentation:** <https://numpy.org/doc/>
8. **Joblib documentation:** <https://joblib.readthedocs.io/>
9. **OpenAI ChatGPT:** <https://chatgpt.com/> (development assistance; see the AI-use disclosure above).

If additional external code, images, datasets, or text are incorporated later, add a source link and describe the use here or alongside the relevant code.

## Privacy and security

The repository is intended to contain source code, model artifacts, sample/demo data, and experiment results only. Do not commit personal or confidential transaction data, account information, passwords, API keys, access tokens, `.env` files, or other secrets. Keep local datasets and private credentials outside Git and maintain the `.gitignore` rules accordingly.

## Disclaimer

SmartShield is an educational machine-learning prototype. It is not financial advice, does not make a legally binding fraud determination, and should not be used as the sole basis for blocking transactions or taking action against customers.
