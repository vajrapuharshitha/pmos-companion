# PMOS Companion: explainable screening, criteria checker and lifestyle guide
PMOS (polyendocrine metabolic ovarian syndrome, formerly PCOS) was renamed in May 2026. Diagnosis still follows the Rotterdam criteria (2 of 3), and many cases go undiagnosed. This project explores an explainable screening aid.

**Screening and education only. Not a diagnosis.**

## Parts
| Folder | What |
|---|---|
| `index.html` | Interactive companion: symptoms, lifestyle, body-systems view (illustrative) |
| `app/pmos_app.py` | Streamlit criteria checker + doctor-ready summary + optional ML slot |
| `ml/train.py` | Leakage-aware training and validation pipeline |

## Method
- Stratified 80/20 split, repeated 5-fold CV for model selection (logistic regression vs gradient boosting)
- **Two experiments:** `full` and `criteria_blind`. Dataset labels come from diagnostic criteria, so using criteria features is partly circular. The blind run tests for signal beyond them.
- Metrics: ROC-AUC with bootstrap 95% CI, PR-AUC, Brier score, sensitivity/specificity, subgroup AUC (BMI ≤25 vs >25)
- Explainability: SHAP summary plots

## Results
Run `python ml/train.py data/pcos.csv` and paste values from `ml/metrics.json` here. (Do not report numbers you have not reproduced.)

## Limitations
Small single-source dataset, no external validation, labels derived from criteria, possible population bias. Prospective and multi-site validation needed before any clinical use.

## Run
`pip install -r requirements.txt` then `streamlit run app/pmos_app.py`.
