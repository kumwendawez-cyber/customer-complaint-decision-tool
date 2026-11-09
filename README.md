# Customer Complaint Prioritization and Resolution Decision-Support Tool (Week 11)

Course AA-5960-02 · Srinivas Chary Kasoju and Wezzie Kumwenda · Decision-maker: Customer Complaints Manager

The tool estimates two risks per complaint to help the manager decide where to look first:

| KPI | Target | Review threshold |
|---|---|---|
| Escalation Risk | `actual_escalation` = 1 | 35% |
| Routing Change Risk | final department ≠ recommended department | 50% |

It supports, and never replaces, the manager's judgment.
   ## Version history
   - **Week_10/**: original Week 10 submission, unchanged.
   - **Main folder (Week 11)**: updated application and evaluation (chronological split, baselines, error analysis, Random Forest and Balanced Logistic comparison, what-if comparison with input checks, reliability tab) and the revised report.

## Folder contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit application (4 tabs: review, what-if, reliability, data/method) |
| `app_utils.py` | Predictions, individual explanations, input validation, automatic summary |
| `pipeline.py` | Data preparation, predictors, model definitions, metrics |
| `train.py` | Compares models on 2025 data, trains the final models, runs the 2026 test, writes `models/` |
| `models/` | `final_models.joblib` and `metrics.json` (created by `train.py`) |
| `Data/` | CSV input files (see "Data access") |
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

## Synthetic Data Generation and Validation

### 1. Original Source Data

The project uses complaint information from the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database.

The original CFPB CSV file is the input to the data-generation script. It is different from the five generated application CSV files stored in the `Data/` folder.

The generator uses the original complaint records and creates synthetic operational information for the decision-support simulation.

### 2. Generate the Synthetic Database

Required Python packages are listed in `requirements.txt`.

Run the generator using the path to the original CFPB CSV file:

```bash
python generate_dataset.py "path/to/cfpb_complaints_raw.csv"
```

The generator creates synthetic customer records, complaint timestamps, department capacity, decision context, predictions, manager decisions, workflow events, and outcomes.

The random seed is fixed to support reproducibility.

### 3. Generated Files

The generator creates:

- `customer_complaints.db` – SQLite database containing nine related tables.
- `database_schema.sql` – SQL database structure.
- `database_validation_summary.txt` – Database validation summary.
- `customer_complaints_project/` – CSV exports of all nine database tables and the main database files.
- `customer_complaints_project_complete.zip` – Complete export package.

The five CSV files in `Data/` are the application inputs. They should not be confused with the original CFPB source file.

### 4. Independent Data Validation

The project includes a separate validation script, `validate_data.py`, to check database integrity and logical consistency.

Run:

```bash
python validate_data.py customer_complaints.db
```

The independent validation completed 24 checks with zero failures on the generated database.

Checks include:

- Required tables and SQLite database integrity.
- Primary-key and foreign-key consistency.
- Complaint-level record counts.
- Complaint, prediction, and decision timestamps.
- Workflow event chronology and status consistency.
- Escalation and resolution consistency.
- Final department and resolution-time consistency.
- Historical capacity availability.
- Prediction probability ranges and nonnegative values.

### 5. Data Limitations

The project combines CFPB complaint information with synthetic operational and outcome variables. Customer identifiers, staffing, workload, capacity, predictions, manager decisions, workflow events, resolution times, and handling costs are simulated.

Complaint narratives were not available in the source data and were retained as missing rather than artificially generated.

The dataset is designed for an academic decision-support prototype. Its validation demonstrates structural and logical consistency, not independent real-world statistical validity or operational performance.

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
