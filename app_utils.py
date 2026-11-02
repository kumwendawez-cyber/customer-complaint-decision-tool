"""Prediction, explanation, input-validation and summary logic for the app (no Streamlit code here)."""
import numpy as np
import pandas as pd
import joblib

LABELS = {
    "prior_complaint_count": "Prior complaints by this customer",
    "known_financial_impact": "Known financial impact",
    "current_workload": "Current workload",
    "available_capacity": "Available capacity",
    "escalation_probability": "Earlier escalation score",
    "issue": "Complaint issue",
    "submission_channel": "Submission channel",
    "predicted_urgency": "Predicted urgency",
    "recommended_department_id": "Recommended department",
}
KPI_NAMES = {"escalation": "Escalation Risk", "routing": "Routing Change Risk"}
ADJUSTABLE = ["current_workload", "available_capacity", "recommended_department_id"]


def load_artifact(path="models/final_models.joblib"):
    return joblib.load(path)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def to_frame(row, art, key):
    return pd.DataFrame([{c: row.get(c, np.nan) for c in art["kpis"][key]["features"]}])


def predict(art, key, row):
    """Predicted probability for one complaint (dict of feature values)."""
    return float(art["models"][key].predict_proba(to_frame(row, art, key))[0, 1])


def predict_many(art, key, df):
    return art["models"][key].predict_proba(df[art["kpis"][key]["features"]])[:, 1]


def _group(name, cat_cols):
    n = name.split("__", 1)[1]
    if n.startswith("missingindicator_"):
        return n.replace("missingindicator_", "")
    for c in cat_cols:
        if n.startswith(c + "_"):
            return c
    return n


def explain(art, key, row):
    """Individual explanation: how each variable moves THIS complaint's predicted risk
    compared with a typical complaint (percentage points). Based on the logistic model's
    own arithmetic; it describes the model, not real-world cause and effect."""
    k, model = art["kpis"][key], art["models"][key]
    pre, clf = model.named_steps["preprocessor"], model.named_steps["classifier"]
    x = pre.transform(to_frame(row, art, key))
    x = (x.toarray() if hasattr(x, "toarray") else np.asarray(x))[0]
    contrib = clf.coef_[0] * (x - k["train_mean"])
    logit = float(clf.intercept_[0] + clf.coef_[0] @ x)
    totals = {}
    for name, c in zip(k["feature_names"], contrib):
        g = _group(name, k["cat"])
        totals[g] = totals.get(g, 0.0) + float(c)
    p_full = _sigmoid(logit)
    rows = []
    for g, c in totals.items():
        pp = (p_full - _sigmoid(logit - c)) * 100
        direction = "Raises risk" if pp > 0.05 else ("Lowers risk" if pp < -0.05 else "Little effect")
        rows.append({"Factor": LABELS.get(g, g), "Effect (pct. points)": round(pp, 1), "Direction": direction})
    df = pd.DataFrame(rows)
    return df.reindex(df["Effect (pct. points)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def expected_capacity(art, workload, window=0.4):
    """Capacity values that have actually occurred in history at a similar workload."""
    pairs = art["wc_pairs"]
    sel = pairs[np.abs(pairs[:, 0] - workload) <= window][:, 1]
    if len(sel) < 20:
        return None
    return dict(n=len(sel), lo=float(sel.min()), hi=float(sel.max()),
                p5=float(np.percentile(sel, 5)), p95=float(np.percentile(sel, 95)))


def validate(art, scenario, baseline):
    """Return (errors, warnings). Errors block the prediction; warnings do not."""
    errors, warnings = [], []
    r = art["ranges"]
    wl, cap = scenario["current_workload"], scenario["available_capacity"]
    if wl is None or cap is None or pd.isna(wl) or pd.isna(cap):
        return ["Workload and capacity must both be filled in."], []
    if cap != int(cap):
        errors.append("Available capacity must be a whole number.")
    lo, hi = r["current_workload"]
    if not (lo <= wl <= hi):
        errors.append(f"Workload {wl:g} is outside the range seen in past data ({lo:g}–{hi:g}). "
                      "The model has no experience there, so no prediction is shown.")
    lo, hi = r["available_capacity"]
    if not (lo <= cap <= hi):
        errors.append(f"Capacity {cap:g} is outside the range seen in past data ({lo:g}–{hi:g}). "
                      "No prediction is shown.")
    if not errors:
        band = expected_capacity(art, wl)
        if band is None:
            errors.append(f"Workload {wl:g} is too rare in past data to give a reliable prediction.")
        elif not (band["lo"] - 3 <= cap <= band["hi"] + 3):
            errors.append(f"A workload of {wl:g} has only occurred with capacity between {band['lo']:g} and "
                          f"{band['hi']:g}. Capacity {cap:g} is an implausible combination, so no prediction "
                          "is shown. Change both together (more capacity normally goes with lower workload).")
        elif not (band["p5"] - 1 <= cap <= band["p95"] + 1):
            warnings.append(f"Unusual combination: at a workload near {wl:g}, capacity is normally between "
                            f"{band['p5']:g} and {band['p95']:g}. Treat this result with extra caution.")
    d = scenario.get("recommended_department_id")
    if d not in art["categories"]["recommended_department_id"]:
        errors.append("Recommended department must be one of the departments the tool can recommend.")
    return errors, warnings


def changed_inputs(baseline, scenario):
    out = []
    for c in ADJUSTABLE:
        if baseline[c] != scenario[c]:
            out.append((LABELS[c], baseline[c], scenario[c]))
    return out


def fmt(c, v):
    return f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def build_summary(art, baseline, scenarios, base_pred):
    """Plain-language summary. scenarios = {name: (row, preds or None)}; base_pred = {kpi: prob}."""
    thr = {k: art["kpis"][k]["threshold"] for k in KPI_NAMES}
    lines = []
    valid = {n: v for n, v in scenarios.items() if v[1] is not None}
    if not valid:
        return "No valid scenario to compare yet. Fix the input messages above to see a comparison."
    for name, (row, preds) in valid.items():
        ch = changed_inputs(baseline, row)
        if not ch:
            lines.append(f"**{name}** has no input changes, so predictions equal the baseline.")
            continue
        what = "; ".join(f"{lab} {fmt(None, a)} → {fmt(None, b)}" for lab, a, b in ch)
        parts, moves = [], {}
        for k, nm in KPI_NAMES.items():
            d = (preds[k] - base_pred[k]) * 100
            moves[k] = d
            before, after = base_pred[k] >= thr[k], preds[k] >= thr[k]
            flag = ("and the review flag does not change" if before == after else
                    ("and it now crosses the review threshold (flagged)" if after
                     else "and it now falls below the review threshold (not flagged)"))
            parts.append(f"{nm} moves from {base_pred[k]*100:.1f}% to {preds[k]*100:.1f}% "
                         f"({d:+.1f} percentage points), {flag}")
        lines.append(f"**{name}** ({what}): " + "; ".join(parts) + ".")
        if moves["escalation"] * moves["routing"] < 0 and min(abs(moves["escalation"]), abs(moves["routing"])) >= 0.5:
            lines.append(f"Trade-off in {name}: the two risks move in opposite directions, "
                         "so improving one does not improve the other.")
        if max(abs(v) for v in moves.values()) < 2:
            lines.append(f"In {name} the changes are small (under 2 percentage points). Given the model's "
                         "moderate accuracy, this should be read as little to no meaningful difference.")
    lines.append("**How to read this:** these are model-based estimates of how the predictions change when "
                 "inputs change. They are not proven cause-and-effect and not guaranteed outcomes. The model has "
                 "only moderate predictive ability (ROC-AUC about 0.64–0.66). Workload and capacity have very "
                 "little influence in this model, so use these results alongside your own judgment.")
    return "\n\n".join(lines)
