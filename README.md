# Customer Complaint Prioritization and Resolution Decision-Support Tool (Week 11)

Course AA-5960-02 · Srinivas Chary Kasoju and Wezzie Kumwenda · Decision-maker: Customer Complaints Manager

The tool estimates two risks per complaint to help the manager decide where to look first:

| KPI | Target | Review threshold |
|---|---|---|
| Escalation Risk | `actual_escalation` = 1 | 35% |
| Routing Change Risk | final department ≠ recommended department | 50% |

It supports, and never replaces, the manager's judgment.

## Folder contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit application (4 tabs: review, what-if, reliability, data/method) |
| `app_utils.py` | Predictions, individual explanations, input validation, automatic summary |
| `pipeline.py` | Data preparation, predictors, model definitions, metrics |
| `train.py` | Compares models on 2025 data, trains the final models, runs the 2026 test, writes `models/` |
| `models/` | `final_models.joblib` and `metrics.json` (created by `train.py`) |
| `data/` | CSV input files (see "Data access") |
| `requirements.txt` | Dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the five CSV files in ./Data (see below)
python train.py            # ~1 minute; recreates models/ (required if your scikit-learn version differs)
streamlit run app.py
```

To keep the data elsewhere, set `COMPLAINT_DATA_DIR=/path/to/folder` before running either command.

## Data access

Required files in `Data/`: `complaints.csv`, `decision_context.csv`, `predictions.csv`, `outcomes.csv`, `departments.csv`.
Some operational and outcome variables are synthetic. **[Team: state here which files are included in the repository and, for any restricted file, who to contact and how to request access.]**
Never commit restricted or confidential data to a public repository.

## How the data is split (chronological, by `date_received` year)

| Stage | Years | Rows (escalation / routing) | Used for |
|---|---|---|---|
| Training | 2023–2024 | 5,495 / 5,495 | Fit models for comparison |
| Development | 2025 | 4,897 / 4,897 | Compare Logistic, Balanced Logistic, Random Forest; choose the model |
| Final training | 2023–2025 | 10,392 / 10,392 | Refit the retained model |
| Final test | 2026 | 4,284 / 4,281 | Evaluated once; never used for selection or thresholds |

A time-based split is used because the tool predicts for future complaints from past ones; a random split would leak future information.
The 35% and 50% thresholds were fixed before the 2026 evaluation.

## Predictors and leakage

Escalation: prior complaint count, known financial impact, current workload, available capacity, earlier escalation score, issue, submission channel, predicted urgency.
Routing: the same plus recommended department.
`actual_escalation` and `final_department_id` are used only to build targets. The earlier escalation score, predicted urgency and recommended department come from `predictions.csv` (`complaint_model_v1`), stamped a median of 2.7 minutes (max 5) after the complaint arrived and on or before the close date for all closed complaints.

## Results (2026 final test)

| KPI | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Escalation | Majority baseline | 0.538 | 0.000 | 0.000 | 0.000 | 0.500 |
| Escalation | Logistic Regression | 0.555 | 0.511 | 0.863 | 0.642 | 0.644 |
| Routing | Majority baseline | 0.595 | 0.595 | 1.000 | 0.746 | 0.500 |
| Routing | Logistic Regression | 0.628 | 0.651 | 0.807 | 0.721 | 0.658 |

Model comparison on 2025 development data is in `models/metrics.json` and in the app (tab 3). Logistic Regression was retained over Balanced Logistic Regression and Random Forest.

## Known limitations

Moderate predictive ability (ROC-AUC about 0.64–0.66); many false positives; Escalation Risk relies mainly on an earlier model's score; workload and capacity have very little influence; some data is synthetic; no confidence intervals; what-if results are model-based, not causal.
