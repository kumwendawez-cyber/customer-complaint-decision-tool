import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Complaint Decision-Support Tool",
    page_icon="📋",
    layout="wide"
)

st.title("Customer Complaint Prioritization and Resolution Decision-Support Tool")

st.caption(
    "Week 10 Prototype — Two Predictive KPIs, Model Interpretation, and What-If Analysis"
)

APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    complaint_file = APP_DIR / "complaints.csv"
    decision_context_file = APP_DIR / "decision_context.csv"
    predictions_file = APP_DIR / "predictions.csv"
    departments_file = APP_DIR / "departments.csv"
    outcomes_file = APP_DIR / "outcomes.csv"

    required_files = [
        complaint_file,
        decision_context_file,
        predictions_file,
        departments_file,
        outcomes_file,
    ]

    missing_files = [p.name for p in required_files if not p.exists()]

    if missing_files:
        st.error(
            "Missing required file(s): "
            + ", ".join(missing_files)
            + ". Add them to the same GitHub folder as app.py."
        )
        st.stop()

    complaints = pd.read_csv(
        complaint_file,
        dtype={"complaint_id": str}
    )

    decision_context = pd.read_csv(
        decision_context_file,
        dtype={"complaint_id": str}
    )

    predictions = pd.read_csv(
        predictions_file,
        dtype={"complaint_id": str}
    )

    departments = pd.read_csv(
        departments_file,
        dtype={"department_id": str}
    )

    outcomes = pd.read_csv(
        outcomes_file,
        dtype={"complaint_id": str, "final_department_id": str}
    )

    # Clean IDs
    complaints["complaint_id"] = complaints["complaint_id"].str.strip()
    decision_context["complaint_id"] = decision_context["complaint_id"].str.strip()
    predictions["complaint_id"] = predictions["complaint_id"].str.strip()
    outcomes["complaint_id"] = outcomes["complaint_id"].str.strip()
    outcomes["final_department_id"] = outcomes["final_department_id"].str.strip()

    # Dates
    complaints["date_received"] = pd.to_datetime(
        complaints["date_received"],
        errors="coerce"
    )

    # Department lookup
    department_lookup = departments.rename(
        columns={
            "department_id": "recommended_department_id",
            "department_name": "recommended_department_name"
        }
    )

    # -----------------------------------------------------
    # JOIN TABLES
    # -----------------------------------------------------

    data = complaints.merge(
        decision_context[
            [
                "complaint_id",
                "prior_complaint_count",
                "known_financial_impact",
                "current_workload",
                "available_capacity"
            ]
        ],
        on="complaint_id",
        how="left"
    )

    data = data.merge(
        predictions[
            [
                "complaint_id",
                "recommended_department_id",
                "predicted_urgency",
                "urgency_probability",
                "escalation_probability"
            ]
        ],
        on="complaint_id",
        how="left"
    )

    data = data.merge(
        outcomes[
            [
                "complaint_id",
                "final_department_id",
                "actual_escalation"
            ]
        ],
        on="complaint_id",
        how="left"
    )

    data = data.merge(
        department_lookup,
        on="recommended_department_id",
        how="left"
    )

    data["recommended_department"] = (
        data["recommended_department_id"].fillna("Unknown")
        + " - "
        + data["recommended_department_name"].fillna("Unknown")
    )

    # Sort High -> Medium -> Low
    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    data["priority_order"] = (
        data["predicted_urgency"]
        .map(priority_order)
        .fillna(9)
    )

    data = data.sort_values(
        ["priority_order", "urgency_probability"],
        ascending=[True, False]
    )

    return data, departments


data, departments = load_data()

# ---------------------------------------------------------
# WEEK 10 FINAL MODELS
# ---------------------------------------------------------

ESC_NUMERIC = [
    "prior_complaint_count",
    "known_financial_impact",
    "current_workload",
    "available_capacity",
    "escalation_probability",
]

ESC_CATEGORICAL = [
    "issue",
    "submission_channel",
    "predicted_urgency",
]

ESC_FEATURES = ESC_NUMERIC + ESC_CATEGORICAL

ROUTE_NUMERIC = ESC_NUMERIC.copy()
ROUTE_CATEGORICAL = ESC_CATEGORICAL + [
    "recommended_department_id"
]
ROUTE_FEATURES = ROUTE_NUMERIC + ROUTE_CATEGORICAL

ESC_THRESHOLD = 0.35
ROUTE_THRESHOLD = 0.50

# These are the final held-out 2026 permutation-importance results
# produced in the Week 10 modeling notebook.
ESC_GLOBAL_IMPORTANCE = pd.DataFrame({
    "Predictor": [
        "escalation_probability",
        "predicted_urgency",
        "prior_complaint_count",
        "available_capacity",
        "submission_channel",
        "current_workload",
        "known_financial_impact",
        "issue",
    ],
    "Importance": [
        0.085891,
        0.012111,
        0.005530,
        0.000365,
        0.000311,
        0.000010,
        -0.001020,
        -0.001277,
    ],
})

ROUTE_GLOBAL_IMPORTANCE = pd.DataFrame({
    "Predictor": [
        "escalation_probability",
        "recommended_department_id",
        "predicted_urgency",
        "prior_complaint_count",
        "available_capacity",
        "submission_channel",
        "known_financial_impact",
        "current_workload",
        "issue",
    ],
    "Importance": [
        0.063193,
        0.059587,
        0.008915,
        0.004728,
        0.001710,
        0.000698,
        -0.000116,
        -0.000250,
        -0.000501,
    ],
})


def make_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


@st.cache_resource
def train_final_models(model_data):
    model_data = model_data.copy()
    model_data["year"] = model_data["date_received"].dt.year

    # -----------------------------
    # KPI 1 — Escalation Risk
    # -----------------------------
    esc = model_data.dropna(
        subset=["actual_escalation", "date_received"]
    ).copy()
    esc["actual_escalation"] = esc["actual_escalation"].astype(int)

    esc_train = esc[esc["year"] <= 2025].copy()
    esc_test = esc[esc["year"] == 2026].copy()

    escalation_model = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    ESC_NUMERIC,
                    ESC_CATEGORICAL,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )

    escalation_model.fit(
        esc_train[ESC_FEATURES],
        esc_train["actual_escalation"],
    )

    esc_prob = escalation_model.predict_proba(
        esc_test[ESC_FEATURES]
    )[:, 1]
    esc_pred = (esc_prob >= ESC_THRESHOLD).astype(int)
    esc_y = esc_test["actual_escalation"]
    esc_cm = confusion_matrix(esc_y, esc_pred)

    esc_metrics = {
        "accuracy": accuracy_score(esc_y, esc_pred),
        "precision": precision_score(esc_y, esc_pred),
        "recall": recall_score(esc_y, esc_pred),
        "f1": f1_score(esc_y, esc_pred),
        "roc_auc": roc_auc_score(esc_y, esc_prob),
        "threshold": ESC_THRESHOLD,
        "test_rows": len(esc_test),
        "confusion_matrix": esc_cm,
    }

    # -----------------------------
    # KPI 2 — Routing Change Risk
    # -----------------------------
    route = model_data.dropna(
        subset=[
            "recommended_department_id",
            "final_department_id",
            "date_received",
        ]
    ).copy()

    route["routing_change"] = (
        route["recommended_department_id"].astype(str)
        != route["final_department_id"].astype(str)
    ).astype(int)

    route_train = route[route["year"] <= 2025].copy()
    route_test = route[route["year"] == 2026].copy()

    routing_model = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    ROUTE_NUMERIC,
                    ROUTE_CATEGORICAL,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )

    routing_model.fit(
        route_train[ROUTE_FEATURES],
        route_train["routing_change"],
    )

    route_prob = routing_model.predict_proba(
        route_test[ROUTE_FEATURES]
    )[:, 1]
    route_pred = (route_prob >= ROUTE_THRESHOLD).astype(int)
    route_y = route_test["routing_change"]
    route_cm = confusion_matrix(route_y, route_pred)

    route_metrics = {
        "accuracy": accuracy_score(route_y, route_pred),
        "precision": precision_score(route_y, route_pred),
        "recall": recall_score(route_y, route_pred),
        "f1": f1_score(route_y, route_pred),
        "roc_auc": roc_auc_score(route_y, route_prob),
        "threshold": ROUTE_THRESHOLD,
        "test_rows": len(route_test),
        "confusion_matrix": route_cm,
    }

    return {
        "escalation_model": escalation_model,
        "routing_model": routing_model,
        "escalation_metrics": esc_metrics,
        "routing_metrics": route_metrics,
        "escalation_test": esc_test,
        "routing_test": route_test,
    }


MODEL_BUNDLE = train_final_models(data)


def get_logistic_contributions(
    model,
    X_row,
    numeric_features,
    categorical_features,
):
    """Aggregate logistic-regression feature contributions to original predictors."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(X_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed).reshape(-1)

    transformed_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    contributions = transformed * coefficients

    detail = pd.DataFrame(
        {
            "Transformed_Feature": transformed_names,
            "Contribution": contributions,
        }
    )

    def original_predictor(name):
        raw = name.split("__", 1)[-1]
        if raw.startswith("missingindicator_"):
            raw = raw.replace("missingindicator_", "", 1)
        if raw in numeric_features:
            return raw
        for col in sorted(
            categorical_features,
            key=len,
            reverse=True,
        ):
            if raw == col or raw.startswith(col + "_"):
                return col
        return raw

    detail["Predictor"] = detail["Transformed_Feature"].apply(
        original_predictor
    )

    grouped = (
        detail.groupby("Predictor", as_index=False)["Contribution"]
        .sum()
    )
    grouped["Magnitude"] = grouped["Contribution"].abs()
    grouped["Direction"] = np.where(
        grouped["Contribution"] > 0.001,
        "Raises predicted risk",
        np.where(
            grouped["Contribution"] < -0.001,
            "Lowers predicted risk",
            "Little effect",
        ),
    )

    return grouped.sort_values("Magnitude", ascending=False)


def prediction_summary(
    model_name,
    baseline_probability,
    updated_probability,
    threshold,
    changes,
):
    difference = updated_probability - baseline_probability

    if difference > 0.001:
        change_text = (
            f"increased by {difference * 100:.1f} percentage points"
        )
    elif difference < -0.001:
        change_text = (
            f"decreased by {abs(difference) * 100:.1f} percentage points"
        )
    else:
        change_text = "changed very little"

    threshold_text = (
        "above" if updated_probability >= threshold else "below"
    )

    changed_inputs = ", ".join(changes) if changes else "no inputs"

    return (
        f"The baseline {model_name.lower()} prediction was "
        f"{baseline_probability * 100:.1f}%. After changing {changed_inputs}, "
        f"the prediction became {updated_probability * 100:.1f}%. "
        f"The model-based predicted risk {change_text} and is {threshold_text} "
        f"the {threshold * 100:.0f}% review threshold. This is a model-based "
        "scenario, not evidence that changing these inputs would cause the "
        "real-world outcome to change by the same amount."
    )

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Complaint Queue"

if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

if "demo_decisions" not in st.session_state:
    st.session_state.demo_decisions = {}


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Navigation")

screen_options = [
    "Complaint Queue",
    "Complaint Review",
    "Model Performance",
    "What-If Analysis"
]

if "navigation_radio" not in st.session_state:
    st.session_state.navigation_radio = (
        st.session_state.current_screen
    )


def sidebar_screen_changed():
    st.session_state.current_screen = (
        st.session_state.navigation_radio
    )


def open_selected_complaint():
    st.session_state.selected_id = (
        st.session_state.queue_selected_id
    )
    st.session_state.current_screen = "Complaint Review"
    st.session_state.navigation_radio = "Complaint Review"


def go_to_queue():
    st.session_state.current_screen = "Complaint Queue"
    st.session_state.navigation_radio = "Complaint Queue"


screen = st.sidebar.radio(
    "Select Screen",
    screen_options,
    key="navigation_radio",
    on_change=sidebar_screen_changed
)

st.session_state.current_screen = screen

st.sidebar.divider()

st.sidebar.info(
    "The model provides recommendations and risk estimates. "
    "The Customer Complaints Manager keeps the final decision."
)


# =========================================================
# COMPLAINT QUEUE
# =========================================================

if screen == "Complaint Queue":

    st.header("Complaint Queue")

    st.write(
        "This screen helps the manager identify which complaints "
        "should be reviewed first."
    )

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_priority = st.multiselect(
            "Priority",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )

    with col2:

        issue_options = sorted(
            data["issue"]
            .dropna()
            .unique()
        )

        selected_issue = st.multiselect(
            "Issue",
            issue_options
        )

    with col3:

        department_options = sorted(
            data["recommended_department"]
            .dropna()
            .unique()
        )

        selected_department = st.multiselect(
            "Recommended Department",
            department_options
        )

    filtered = data.copy()

    filtered = filtered[
        filtered["predicted_urgency"]
        .isin(selected_priority)
    ]

    if selected_issue:

        filtered = filtered[
            filtered["issue"]
            .isin(selected_issue)
        ]

    if selected_department:

        filtered = filtered[
            filtered["recommended_department"]
            .isin(selected_department)
        ]

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    total = len(filtered)

    high_priority = (
        filtered["predicted_urgency"] == "High"
    ).sum()

    average_escalation = (
        filtered["escalation_probability"].mean()
        * 100
    )

    average_capacity = (
        filtered["available_capacity"].mean()
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Complaints Shown",
        f"{total:,}"
    )

    metric2.metric(
        "High Priority",
        f"{high_priority:,}"
    )

    metric3.metric(
        "Average Escalation Probability",
        f"{average_escalation:.1f}%"
    )

    metric4.metric(
        "Average Available Capacity",
        f"{average_capacity:.1f}"
    )

    st.divider()

    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

    display_data = filtered[
        [
            "complaint_id",
            "date_received",
            "issue",
            "sub_issue",
            "predicted_urgency",
            "urgency_probability",
            "recommended_department",
            "escalation_probability",
            "current_workload",
            "available_capacity"
        ]
    ].copy()

    display_data["urgency_probability"] = (
        display_data["urgency_probability"]
        * 100
    ).round(1)

    display_data["escalation_probability"] = (
        display_data["escalation_probability"]
        * 100
    ).round(1)

    display_data["current_workload"] = (
        display_data["current_workload"]
        .round(2)
    )

    display_data = display_data.rename(
        columns={
            "complaint_id": "Complaint ID",
            "date_received": "Date Received",
            "issue": "Issue",
            "sub_issue": "Sub-Issue",
            "predicted_urgency": "Priority",
            "urgency_probability": "Urgency %",
            "recommended_department": "Recommended Department",
            "escalation_probability": "Escalation %",
            "current_workload": "Current Workload",
            "available_capacity": "Available Capacity"
        }
    )

    st.dataframe(
        display_data.head(200),
        use_container_width=True,
        hide_index=True,
        height=450
    )

    if len(display_data) > 200:

        st.caption(
            "Showing the first 200 complaints."
        )

    # -----------------------------------------------------
    # SELECT COMPLAINT
    # -----------------------------------------------------

    st.subheader("Open Complaint")

    complaint_ids = (
        filtered["complaint_id"]
        .astype(str)
        .tolist()
    )

    if complaint_ids:

        default_index = 0

        if (
            st.session_state.selected_id
            in complaint_ids
        ):
            default_index = complaint_ids.index(
                st.session_state.selected_id
            )

        st.selectbox(
            "Select Complaint ID",
            complaint_ids,
            index=default_index,
            key="queue_selected_id"
        )

        st.button(
            "Review Selected Complaint",
            type="primary",
            on_click=open_selected_complaint
        )

    else:

        st.warning(
            "No complaints match the selected filters."
        )
# =========================================================
# COMPLAINT REVIEW
# =========================================================

elif screen == "Complaint Review":

    st.header(
        "Complaint Review and Manager Decision"
    )

    all_ids = (
        data["complaint_id"]
        .astype(str)
        .tolist()
    )

    if (
        st.session_state.selected_id
        not in all_ids
    ):

        st.session_state.selected_id = (
            all_ids[0]
        )

    selected_id = st.selectbox(
        "Complaint ID",
        all_ids,
        index=all_ids.index(
            st.session_state.selected_id
        )
    )

    st.session_state.selected_id = (
        selected_id
    )

    record = data[
        data["complaint_id"]
        .astype(str)
        == selected_id
    ].iloc[0]

    # -----------------------------------------------------
    # THREE INFORMATION PANELS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # ------------------
    # Complaint Details
    # ------------------

    with col1:

        st.subheader("Complaint Details")

        st.write(
            "**Complaint ID:**",
            record["complaint_id"]
        )

        st.write(
            "**Date Received:**",
            record["date_received"].date()
        )

        st.write(
            "**Product:**",
            record["product"]
        )

        st.write(
            "**Sub-product:**",
            record["sub_product"]
        )

        st.write(
            "**Issue:**",
            record["issue"]
        )

        if pd.notna(
            record["sub_issue"]
        ):

            st.write(
                "**Sub-Issue:**",
                record["sub_issue"]
            )

        else:

            st.write(
                "**Sub-Issue:** Not Available"
            )

        st.write(
            "**Submission Channel:**",
            record["submission_channel"]
        )

    # ------------------
    # Decision Context
    # ------------------

    with col2:

        st.subheader("Decision Context")

        st.metric(
            "Prior Complaints",
            int(
                record[
                    "prior_complaint_count"
                ]
            )
        )

        if pd.notna(
            record[
                "known_financial_impact"
            ]
        ):

            st.metric(
                "Known Financial Impact",
                f"${record['known_financial_impact']:,.2f}"
            )

        else:

            st.metric(
                "Known Financial Impact",
                "Not Available"
            )

        st.metric(
            "Current Workload",
            f"{record['current_workload']:.2f}"
        )

        st.metric(
            "Available Capacity",
            int(
                record[
                    "available_capacity"
                ]
            )
        )

    # ------------------
    # Model Recommendation
    # ------------------

    with col3:

        st.subheader(
            "Model Recommendation"
        )

        st.metric(
            "Predicted Priority",
            record[
                "predicted_urgency"
            ]
        )

        st.metric(
            "Urgency Probability",
            f"{record['urgency_probability'] * 100:.1f}%"
        )

        st.metric(
            "Recommended Department",
            record[
                "recommended_department"
            ]
        )

        st.metric(
            "Escalation Probability",
            f"{record['escalation_probability'] * 100:.1f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # MANAGER DECISION
    # -----------------------------------------------------

    st.subheader(
        "Manager Decision"
    )

    st.write(
        "The manager can accept or override "
        "the model recommendation."
    )

    department_dict = dict(
        zip(
            departments[
                "department_id"
            ].astype(str),

            departments[
                "department_name"
            ].astype(str)
        )
    )

    department_ids = list(
        department_dict.keys()
    )

    decision1, decision2, decision3 = (
        st.columns(3)
    )

    with decision1:

        priorities = [
            "Low",
            "Medium",
            "High"
        ]

        default_priority = (
            priorities.index(
                record[
                    "predicted_urgency"
                ]
            )
        )

        final_priority = st.selectbox(
            "Final Priority",
            priorities,
            index=default_priority
        )

    with decision2:

        recommended_id = str(
            record[
                "recommended_department_id"
            ]
        )

        if recommended_id in department_ids:

            default_department = (
                department_ids.index(
                    recommended_id
                )
            )

        else:

            default_department = 0

        final_department = (
            st.selectbox(
                "Final Department",
                department_ids,
                index=default_department,
                format_func=lambda x:
                f"{x} - {department_dict[x]}"
            )
        )

    with decision3:

        escalate = st.radio(
            "Escalate Complaint?",
            [
                "No",
                "Yes"
            ],
            horizontal=True
        )

    reason = st.text_area(
        "Manager Note",
        placeholder=(
            "Optional: explain why "
            "the recommendation was changed."
        )
    )

    if st.button(
        "Confirm Decision",
        type="primary"
    ):

        st.session_state.demo_decisions[
            selected_id
        ] = {

            "priority":
                final_priority,

            "department":
                final_department,

            "escalation":
                escalate,

            "reason":
                reason
        }

        st.success(
            f"""
            Decision saved for Complaint {selected_id}

            Priority: {final_priority}

            Department: {final_department} -
            {department_dict[final_department]}

            Escalate: {escalate}
            """
        )

    # -----------------------------------------------------
    # SHOW SAVED DECISION
    # -----------------------------------------------------

    if (
        selected_id
        in st.session_state.demo_decisions
    ):

        saved = (
            st.session_state
            .demo_decisions[
                selected_id
            ]
        )

        st.info(
            f"""
            Current Demo Decision

            Priority: {saved['priority']}

            Department: {saved['department']} -
            {department_dict[saved['department']]}

            Escalation: {saved['escalation']}
            """
        )

    st.divider()

    st.button(
        "Back to Complaint Queue",
        on_click=go_to_queue
    )


# =========================================================
# KPI & MODEL PERFORMANCE
# =========================================================

elif screen == "Model Performance":

    st.header("KPI & Model Performance")
    st.write(
        "This screen summarizes the two predictive KPIs used by the "
        "Customer Complaints Manager and shows how the final models "
        "performed on the held-out 2026 test period."
    )

    tab1, tab2 = st.tabs([
        "KPI 1 — Escalation Risk",
        "KPI 2 — Routing Change Risk",
    ])

    with tab1:
        metrics = MODEL_BUNDLE["escalation_metrics"]

        st.subheader("Escalation Risk")
        st.write(
            "**KPI definition:** probability that a complaint will eventually escalate."
        )
        st.write(
            "**Why it matters:** helps the manager identify complaints that may "
            "need earlier review or additional attention, supporting operational "
            "performance and reducing the chance of missing higher-risk cases."
        )
        st.caption(
            "Final model: Logistic Regression. The 35% review threshold was selected "
            "during development; the results below are from the untouched 2026 test set."
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        m4.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")
        m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("#### Confusion Matrix")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: No escalation", "Actual: Escalation"],
                columns=["Predicted: No", "Predicted: Yes"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(f"Held-out 2026 test cases: {metrics['test_rows']:,}")

        with right:
            st.markdown("#### Manager interpretation")
            st.write(
                f"At the {metrics['threshold'] * 100:.0f}% review threshold, "
                f"the model identified {metrics['recall'] * 100:.1f}% of complaints "
                "that eventually escalated. Higher recall is useful for screening, "
                "but it also creates more false positives, so the model supports "
                "manager review rather than making an automatic decision."
            )

        st.markdown("#### Overall Predictor Importance")
        importance_display = ESC_GLOBAL_IMPORTANCE.copy()
        importance_display["Importance"] = importance_display["Importance"].round(4)
        st.dataframe(
            importance_display,
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            ESC_GLOBAL_IMPORTANCE.set_index("Predictor")["Importance"]
        )
        st.caption(
            "Global permutation importance measures how much held-out ROC-AUC changes "
            "when a predictor is shuffled. Values near zero, including small negative "
            "values, indicate little useful contribution on the test set. Importance "
            "does not establish causation."
        )

        with st.expander("Model inputs and rationale"):
            st.write(
                "**Predictors:** prior complaint count, known financial impact, "
                "current workload, available capacity, escalation probability, issue, "
                "submission channel, and predicted urgency."
            )
            st.write(
                "**Rationale:** Logistic Regression fits the binary outcome, returns "
                "a probability for decision support, and allows transparent interpretation "
                "of individual prediction contributions."
            )

    with tab2:
        metrics = MODEL_BUNDLE["routing_metrics"]

        st.subheader("Routing Change Risk")
        st.write(
            "**KPI definition:** probability that the initially recommended department "
            "will differ from the department that ultimately resolves the complaint."
        )
        st.write(
            "**Why it matters:** a likely routing change can signal possible transfers, "
            "extra handling effort, or delay. Earlier review can help the manager focus "
            "attention on cases whose initial routing may be less stable."
        )
        st.caption(
            "Final model: Logistic Regression. The 50% classification threshold was "
            "locked before the final 2026 evaluation."
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        m4.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")
        m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("#### Confusion Matrix")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: No routing change", "Actual: Routing change"],
                columns=["Predicted: No", "Predicted: Yes"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(f"Held-out 2026 test cases: {metrics['test_rows']:,}")

        with right:
            st.markdown("#### Manager interpretation")
            st.write(
                f"At the {metrics['threshold'] * 100:.0f}% threshold, the model "
                f"identified {metrics['recall'] * 100:.1f}% of complaints whose final "
                "department differed from the original recommendation. This is a "
                "screening signal for review, not an automatic rerouting rule."
            )

        st.markdown("#### Overall Predictor Importance")
        importance_display = ROUTE_GLOBAL_IMPORTANCE.copy()
        importance_display["Importance"] = importance_display["Importance"].round(4)
        st.dataframe(
            importance_display,
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            ROUTE_GLOBAL_IMPORTANCE.set_index("Predictor")["Importance"]
        )
        st.caption(
            "Global permutation importance describes usefulness across the held-out "
            "test population. It does not mean a predictor causes a routing change."
        )

        with st.expander("Model inputs and rationale"):
            st.write(
                "**Predictors:** the Escalation Risk predictors plus the initially "
                "recommended department."
            )
            st.write(
                "**Rationale:** Logistic Regression provides a probability of routing "
                "change and keeps the model explainable enough for global and local "
                "decision-support interpretation."
            )


# =========================================================
# WHAT-IF ANALYSIS
# =========================================================

elif screen == "What-If Analysis":

    st.header("What-If Analysis")
    st.write(
        "Select an existing complaint as the baseline, modify predictor values, "
        "and compare the updated model prediction with the original prediction."
    )
    st.caption(
        "What-if results are model-based scenarios, not guaranteed causal effects. "
        "Changing an input may or may not materially change the prediction."
    )

    available_ids = data[
        data["date_received"].dt.year == 2026
    ]["complaint_id"].astype(str).tolist()

    if not available_ids:
        available_ids = data["complaint_id"].astype(str).tolist()

    default_idx = 0
    if (
        st.session_state.selected_id is not None
        and str(st.session_state.selected_id) in available_ids
    ):
        default_idx = available_ids.index(str(st.session_state.selected_id))

    selected_what_if_id = st.selectbox(
        "Baseline Complaint ID",
        available_ids,
        index=default_idx,
        key="what_if_complaint_id",
    )

    baseline_record = data[
        data["complaint_id"].astype(str) == selected_what_if_id
    ].iloc[[0]].copy()

    kpi_tab1, kpi_tab2 = st.tabs([
        "Escalation Risk",
        "Routing Change Risk",
    ])

    def scenario_controls(prefix, include_department=False):
        row = baseline_record.iloc[0]

        issue_options = sorted(
            data["issue"].dropna().astype(str).unique().tolist()
        )
        channel_options = sorted(
            data["submission_channel"].dropna().astype(str).unique().tolist()
        )
        urgency_options = ["Low", "Medium", "High"]
        department_options = sorted(
            data["recommended_department_id"].dropna().astype(str).unique().tolist()
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            prior = st.number_input(
                "Prior complaints",
                min_value=0,
                max_value=50,
                value=int(row["prior_complaint_count"]),
                step=1,
                key=f"{prefix}_prior",
            )

            impact_is_missing = pd.isna(row["known_financial_impact"])
            impact_unknown = st.checkbox(
                "Financial impact is not known",
                value=bool(impact_is_missing),
                key=f"{prefix}_impact_unknown",
            )
            impact_default = (
                0.0
                if impact_is_missing
                else float(row["known_financial_impact"])
            )
            financial = st.number_input(
                "Known financial impact ($)",
                min_value=0.0,
                max_value=50000.0,
                value=float(impact_default),
                step=100.0,
                disabled=impact_unknown,
                key=f"{prefix}_financial",
            )

        with c2:
            workload = st.number_input(
                "Current workload",
                min_value=0.0,
                max_value=100.0,
                value=float(row["current_workload"]),
                step=0.25,
                key=f"{prefix}_workload",
            )
            capacity = st.number_input(
                "Available capacity",
                min_value=0,
                max_value=200,
                value=int(row["available_capacity"]),
                step=1,
                key=f"{prefix}_capacity",
            )
            esc_signal = st.slider(
                "Existing escalation signal",
                min_value=0,
                max_value=100,
                value=int(round(float(row["escalation_probability"]) * 100)),
                step=1,
                key=f"{prefix}_esc_signal",
            )

        with c3:
            base_issue = str(row["issue"])
            issue = st.selectbox(
                "Issue",
                issue_options,
                index=issue_options.index(base_issue) if base_issue in issue_options else 0,
                key=f"{prefix}_issue",
            )

            base_channel = str(row["submission_channel"])
            channel = st.selectbox(
                "Submission channel",
                channel_options,
                index=(
                    channel_options.index(base_channel)
                    if base_channel in channel_options
                    else 0
                ),
                key=f"{prefix}_channel",
            )

            base_urgency = str(row["predicted_urgency"])
            urgency = st.selectbox(
                "Predicted urgency",
                urgency_options,
                index=(
                    urgency_options.index(base_urgency)
                    if base_urgency in urgency_options
                    else 1
                ),
                key=f"{prefix}_urgency",
            )

            department = None
            if include_department:
                base_department = str(row["recommended_department_id"])
                department = st.selectbox(
                    "Recommended department",
                    department_options,
                    index=(
                        department_options.index(base_department)
                        if base_department in department_options
                        else 0
                    ),
                    format_func=lambda x: (
                        f"{x} - "
                        + str(
                            departments.loc[
                                departments["department_id"].astype(str) == x,
                                "department_name",
                            ].iloc[0]
                        )
                        if (
                            departments["department_id"].astype(str) == x
                        ).any()
                        else x
                    ),
                    key=f"{prefix}_department",
                )

        return {
            "prior_complaint_count": int(prior),
            "known_financial_impact": (
                np.nan if impact_unknown else float(financial)
            ),
            "current_workload": float(workload),
            "available_capacity": int(capacity),
            "escalation_probability": float(esc_signal) / 100.0,
            "issue": issue,
            "submission_channel": channel,
            "predicted_urgency": urgency,
            **(
                {"recommended_department_id": department}
                if include_department
                else {}
            ),
        }

    def changed_input_names(baseline_row, scenario_row, feature_names):
        changed = []
        for col in feature_names:
            old = baseline_row.iloc[0][col]
            new = scenario_row.iloc[0][col]
            if pd.isna(old) and pd.isna(new):
                continue
            if pd.isna(old) != pd.isna(new):
                changed.append(col)
                continue
            if isinstance(old, (float, np.floating)) or isinstance(new, (float, np.floating)):
                try:
                    if not np.isclose(float(old), float(new), equal_nan=True):
                        changed.append(col)
                except (TypeError, ValueError):
                    if str(old) != str(new):
                        changed.append(col)
            elif str(old) != str(new):
                changed.append(col)
        return changed

    with kpi_tab1:
        st.subheader("KPI 1 — Escalation Risk What-If")

        model = MODEL_BUNDLE["escalation_model"]
        baseline_X = baseline_record[ESC_FEATURES].copy()
        baseline_prob = model.predict_proba(baseline_X)[0, 1]

        st.metric(
            "Baseline escalation risk",
            f"{baseline_prob * 100:.1f}%",
        )

        values = scenario_controls("esc", include_department=False)
        scenario_X = baseline_X.copy()
        for col, value in values.items():
            scenario_X.loc[scenario_X.index[0], col] = value

        updated_prob = model.predict_proba(scenario_X)[0, 1]
        difference = updated_prob - baseline_prob

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Updated escalation risk",
            f"{updated_prob * 100:.1f}%",
            delta=f"{difference * 100:+.1f} pp",
        )
        r2.metric("Review threshold", f"{ESC_THRESHOLD * 100:.0f}%")
        r3.metric(
            "Screening result",
            "Flag for review" if updated_prob >= ESC_THRESHOLD else "Below threshold",
        )

        st.markdown("#### Individual Prediction Contributions")
        contributions = get_logistic_contributions(
            model,
            scenario_X,
            ESC_NUMERIC,
            ESC_CATEGORICAL,
        )
        st.dataframe(
            contributions.round({"Contribution": 4, "Magnitude": 4}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These are local contributions for this complaint. Positive values push "
            "the model toward higher predicted risk; negative values push it lower. "
            "They are model associations, not causal effects."
        )

        changed = changed_input_names(
            baseline_X,
            scenario_X,
            ESC_FEATURES,
        )
        st.markdown("#### Automatically Generated Summary")
        st.info(
            prediction_summary(
                "Escalation Risk",
                baseline_prob,
                updated_prob,
                ESC_THRESHOLD,
                changed,
            )
        )

        with st.expander("Baseline and scenario inputs"):
            comparison = pd.DataFrame(
                {
                    "Predictor": ESC_FEATURES,
                    "Baseline": [baseline_X.iloc[0][c] for c in ESC_FEATURES],
                    "Scenario": [scenario_X.iloc[0][c] for c in ESC_FEATURES],
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)

    with kpi_tab2:
        st.subheader("KPI 2 — Routing Change Risk What-If")

        model = MODEL_BUNDLE["routing_model"]
        baseline_X = baseline_record[ROUTE_FEATURES].copy()
        baseline_prob = model.predict_proba(baseline_X)[0, 1]

        st.metric(
            "Baseline routing change risk",
            f"{baseline_prob * 100:.1f}%",
        )

        values = scenario_controls("route", include_department=True)
        scenario_X = baseline_X.copy()
        for col, value in values.items():
            scenario_X.loc[scenario_X.index[0], col] = value

        updated_prob = model.predict_proba(scenario_X)[0, 1]
        difference = updated_prob - baseline_prob

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Updated routing change risk",
            f"{updated_prob * 100:.1f}%",
            delta=f"{difference * 100:+.1f} pp",
        )
        r2.metric("Review threshold", f"{ROUTE_THRESHOLD * 100:.0f}%")
        r3.metric(
            "Screening result",
            "Flag for review" if updated_prob >= ROUTE_THRESHOLD else "Below threshold",
        )

        st.markdown("#### Individual Prediction Contributions")
        contributions = get_logistic_contributions(
            model,
            scenario_X,
            ROUTE_NUMERIC,
            ROUTE_CATEGORICAL,
        )
        st.dataframe(
            contributions.round({"Contribution": 4, "Magnitude": 4}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These are local contributions for this complaint. They show how this "
            "specific set of predictor values pushes the fitted model's risk estimate "
            "up or down; they do not establish causation."
        )

        changed = changed_input_names(
            baseline_X,
            scenario_X,
            ROUTE_FEATURES,
        )
        st.markdown("#### Automatically Generated Summary")
        st.info(
            prediction_summary(
                "Routing Change Risk",
                baseline_prob,
                updated_prob,
                ROUTE_THRESHOLD,
                changed,
            )
        )

        with st.expander("Baseline and scenario inputs"):
            comparison = pd.DataFrame(
                {
                    "Predictor": ROUTE_FEATURES,
                    "Baseline": [baseline_X.iloc[0][c] for c in ROUTE_FEATURES],
                    "Scenario": [scenario_X.iloc[0][c] for c in ROUTE_FEATURES],
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
