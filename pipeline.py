"""Shared data preparation and model definitions for the Complaint Decision-Support Tool."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

NUMERIC = ["prior_complaint_count", "known_financial_impact", "current_workload",
           "available_capacity", "escalation_probability"]
CAT_ESC = ["issue", "submission_channel", "predicted_urgency"]
CAT_ROUTE = CAT_ESC + ["recommended_department_id"]
ESC_FEATURES = NUMERIC + CAT_ESC
ROUTE_FEATURES = NUMERIC + CAT_ROUTE
ESC_THRESHOLD = 0.35     # fixed in Week 10, before the 2026 evaluation
ROUTE_THRESHOLD = 0.50


def load_data(folder="Data"):
    r = lambda f: pd.read_csv(f"{folder}/{f}.csv")
    return (r("complaints"), r("decision_context"), r("predictions"),
            r("outcomes"), r("departments"))


def build_datasets(folder="Data"):
    complaints, ctx, pred, out, dept = load_data(folder)
    base = (complaints[["complaint_id", "date_received", "issue", "submission_channel"]]
            .merge(ctx[["complaint_id", "prior_complaint_count", "known_financial_impact",
                        "current_workload", "available_capacity"]], on="complaint_id", how="left")
            .merge(pred[["complaint_id", "predicted_urgency", "escalation_probability",
                         "recommended_department_id"]], on="complaint_id", how="left")
            .merge(out[["complaint_id", "actual_escalation", "final_department_id"]],
                   on="complaint_id", how="left"))
    base["date_received"] = pd.to_datetime(base["date_received"], errors="coerce")
    base["year"] = base["date_received"].dt.year
    esc = base.dropna(subset=["actual_escalation"]).copy()
    esc["actual_escalation"] = esc["actual_escalation"].astype(int)
    route = base.dropna(subset=["recommended_department_id", "final_department_id"]).copy()
    route["routing_change"] = (route["recommended_department_id"]
                               != route["final_department_id"]).astype(int)
    return esc, route, dept


def preprocessor(cat):
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scaler", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])


def make_model(kind, cat):
    if kind == "logistic":
        clf = LogisticRegression(max_iter=3000, random_state=42)
    elif kind == "balanced":
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
    elif kind == "forest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                     random_state=42, n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("preprocessor", preprocessor(cat)), ("classifier", clf)])


def metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), F1=f1_score(y, pred), ROC_AUC=roc_auc_score(y, prob))


def build_queue(folder="Data", year=2026):
    """Complaints for the review queue. Outcomes are NOT loaded (they are unknown at review time)."""
    complaints, ctx, pred, out, dept = load_data(folder)
    q = (complaints[["complaint_id", "date_received", "product", "issue", "sub_issue", "submission_channel"]]
         .merge(ctx[["complaint_id", "prior_complaint_count", "known_financial_impact",
                     "current_workload", "available_capacity"]], on="complaint_id", how="left")
         .merge(pred[["complaint_id", "predicted_urgency", "escalation_probability",
                      "recommended_department_id"]], on="complaint_id", how="left"))
    q["date_received"] = pd.to_datetime(q["date_received"], errors="coerce")
    q = q[q["date_received"].dt.year == year].dropna(subset=["recommended_department_id"]).reset_index(drop=True)
    return q
