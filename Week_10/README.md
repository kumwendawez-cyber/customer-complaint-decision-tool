# Customer Complaint Prioritization and Resolution Decision-Support Tool

## Team Project

This repository contains our team project for the Master’s Research Project course at Saint Louis University.

**Project Team**
- Srinivas Chary Kasoju
- Wezzie Kumwenda

**Program:** MS Analytics  
**Course:** AA-5960-02  
**Professor:** Srikanth Mudigonda

---

## Our Project

Our team developed a **Customer Complaint Prioritization and Resolution Decision-Support Tool** for a Customer Complaints Manager at a bank or financial institution.

The purpose of the project is to help the manager make more consistent and informed complaint-handling decisions while keeping the final decision with the human manager.

The application supports four main activities:

- reviewing and prioritizing complaints,
- reviewing model recommendations,
- estimating escalation and routing-change risk,
- testing what-if scenarios and understanding model predictions.

---

## Decision-Maker

The primary user of our tool is a **Customer Complaints Manager**.

The manager needs to decide:

- which complaints need additional attention,
- which complaints may escalate,
- whether an initial department assignment may later change,
- whether model recommendations should be accepted or overridden.

Our application is designed as **decision support**, not as an automated decision-making system.

---

## Predictive KPIs

Our final application uses two predictive KPIs.

### KPI 1 — Escalation Risk

This KPI estimates the probability that a complaint will eventually escalate.

**Final 2026 Test Performance**

| Metric | Result |
| --- | ---: |
| Accuracy | 55.5% |
| Precision | 51.1% |
| Recall | 86.3% |
| F1 Score | 64.2% |
| ROC-AUC | 0.644 |
| Review Threshold | 35% |

We place particular emphasis on **recall** because the model is used as a screening tool. Missing a complaint that later escalates may be more costly than reviewing an additional complaint.

### KPI 2 — Routing Change Risk

This KPI estimates the probability that the initially recommended department will differ from the department that ultimately resolves the complaint.

**Final 2026 Test Performance**

| Metric | Result |
| --- | ---: |
| Accuracy | 62.8% |
| Precision | 65.1% |
| Recall | 80.7% |
| F1 Score | 72.1% |
| ROC-AUC | 0.658 |
| Review Threshold | 50% |

This KPI helps identify complaints whose initial routing may deserve additional managerial review.

---

## Application Features

### Complaint Queue

Allows the manager to review complaints and filter them by priority, issue, and recommended department.

### Complaint Review and Manager Decision

Shows complaint details, operational context, model recommendations, and allows the manager to confirm or override the recommendation.

### KPI & Model Performance

Shows:

- KPI definitions,
- final model performance,
- confusion matrices,
- decision thresholds,
- global predictor importance,
- practical interpretation of model results.

### What-If Analysis

Allows the manager to select an existing complaint as the baseline and modify predictor values.

The application then shows:

- baseline prediction,
- updated prediction,
- change in percentage points,
- review threshold,
- screening result,
- individual predictor contributions,
- automatically generated scenario summary.

The what-if results are **model-based scenarios** and should not be interpreted as guaranteed causal effects.

---

## Modeling Approach

Our team used **Logistic Regression** for both final predictive models.

The models use complaint and operational information available around decision time.

### Escalation Risk Predictors

- prior complaint count
- known financial impact
- current workload
- available capacity
- escalation probability
- issue
- submission channel
- predicted urgency

### Routing Change Risk Predictors

The Routing Change Risk model uses the same predictors plus:

- recommended department

---

## Model Interpretation

To make the models easier for a manager to understand, our application includes both global and local interpretation.

### Overall Predictor Importance

Permutation importance is used to show which predictors are most useful across the held-out test population.

These values describe **predictive usefulness** and do not establish causation.

### Individual Prediction Contributions

For a selected complaint or what-if scenario, the application shows whether each predictor pushes the model prediction higher or lower.

These contributions are model-based associations and should not be interpreted as causal effects.

---

## Validation Design

Our team used a chronological modeling design:

- **2023–2024:** initial training
- **2025:** development and model/threshold selection
- **2023–2025:** final model training
- **2026:** held-out final test period

This structure helps keep final evaluation separate from model development.

---

## Data

The project uses the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database** as the core complaint source, together with synthetic operational and workflow fields created for the project.

The application uses the following main files:

```text
app.py
complaints.csv
decision_context.csv
predictions.csv
departments.csv
outcomes.csv
requirements.txt
```

Additional project files may include:

```text
week10_bootstrap_coefficients.npz
week10_model_metrics.json
escalation_global_importance.csv
routing_global_importance.csv
```

---

## Team Workflow

This project was completed collaboratively as a team.

Our shared work included:

- defining the business problem and decision-maker,
- preparing and validating the project data,
- developing and evaluating predictive models,
- comparing alternative KPI and model options,
- designing and testing the Streamlit decision-support application,
- interpreting model outputs,
- developing what-if analysis,
- preparing project documentation and presentation materials.

A brief member-specific contribution summary is included separately in the final project submission as required by the course.

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application

From the project folder:

```bash
streamlit run app.py
```

The application should open automatically in a browser.

---

## Streamlit Community Cloud Deployment

1. Upload the project files to the team GitHub repository.
2. Make sure `app.py`, `requirements.txt`, and all required CSV files are included.
3. Open Streamlit Community Cloud.
4. Select the GitHub repository.
5. Set `app.py` as the application entry point.
6. Deploy the application.

---

## Important Limitations

- This is an academic research prototype, not a production banking system.
- Some operational and outcome fields are synthetic.
- The predictive models show moderate performance and should not be treated as perfect predictions.
- Predictor importance does not establish causation.
- What-if results are model-based scenarios rather than guaranteed real-world effects.
- The Customer Complaints Manager keeps responsibility for the final decision.

---

## Project Goal

Our team’s goal is to demonstrate how predictive analytics, model interpretation, and interactive what-if analysis can support complaint-management decisions while keeping the human decision-maker in control.
