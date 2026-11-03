"""Train and evaluate the two KPI models, then save everything the app needs.

Run once:  python train.py
Outputs:   models/final_models.joblib  and  models/metrics.json

Procedure (chronological split):
  2023-2024  training for model comparison
  2025       development: compare Logistic / Balanced Logistic / Random Forest
  2023-2025  final training of the retained model
  2026       final test, used ONCE for reporting (never for selection)
"""
import json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from pipeline import (build_datasets, make_model, metrics, NUMERIC, CAT_ESC, CAT_ROUTE,
                      ESC_FEATURES, ROUTE_FEATURES, ESC_THRESHOLD, ROUTE_THRESHOLD)

DATA_DIR = os.environ.get("COMPLAINT_DATA_DIR", "Data")
esc, route, dept = build_datasets(DATA_DIR)
os.makedirs("models", exist_ok=True)

KPIS = {
    "escalation": dict(name="Escalation Risk", df=esc, feat=ESC_FEATURES, cat=CAT_ESC,
                       target="actual_escalation", thr=ESC_THRESHOLD),
    "routing": dict(name="Routing Change Risk", df=route, feat=ROUTE_FEATURES, cat=CAT_ROUTE,
                    target="routing_change", thr=ROUTE_THRESHOLD),
}
MODEL_LABELS = {"logistic": "Original Logistic Regression",
                "balanced": "Balanced Logistic Regression",
                "forest": "Random Forest"}
RETAINED = "logistic"   # chosen on 2025 development data (see dev_comparison)

out = {"split": {}, "dev_comparison": [], "baseline_vs_final": [], "errors": {},
       "importance": {}, "confusion": {}, "flag_rate": {}, "base_rate": {},
       "thresholds": {"escalation": ESC_THRESHOLD, "routing": ROUTE_THRESHOLD},
       "retained_model": MODEL_LABELS[RETAINED]}
artifact = {"models": {}, "kpis": {}}

for key, k in KPIS.items():
    df, feat, cat, tgt, thr = k["df"], k["feat"], k["cat"], k["target"], k["thr"]
    tr, dv = df[df.year <= 2024], df[df.year == 2025]
    fin, te = df[df.year <= 2025], df[df.year == 2026]
    out["split"][key] = dict(train_2023_2024=len(tr), dev_2025=len(dv),
                             final_train_2023_2025=len(fin), test_2026=len(te))

    # ---- 1. Model comparison on 2025 development data only ----
    for kind in ["logistic", "balanced", "forest"]:
        m = make_model(kind, cat).fit(tr[feat], tr[tgt])
        r = metrics(dv[tgt], m.predict_proba(dv[feat])[:, 1], thr)
        out["dev_comparison"].append(dict(KPI=k["name"], Model=MODEL_LABELS[kind], **r))

    # ---- 2. Final model: refit retained model on 2023-2025 ----
    model = make_model(RETAINED, cat).fit(fin[feat], fin[tgt])
    p = model.predict_proba(te[feat])[:, 1]
    pred = (p >= thr).astype(int)

    # ---- 3. Final test (2026), once ----
    base_label = int(round(fin[tgt].mean()))              # majority class from training data
    # majority-class baseline: constant prediction
    bp = np.full(len(te), base_label)
    base_r = dict(Accuracy=accuracy_score(te[tgt], bp),
                  Precision=precision_score(te[tgt], bp, zero_division=0),
                  Recall=recall_score(te[tgt], bp), F1=f1_score(te[tgt], bp, zero_division=0),
                  ROC_AUC=0.5)
    fin_r = metrics(te[tgt], p, thr)
    # single-score baseline: the existing escalation_probability alone (ranking only)
    score_auc = float(roc_auc_score(te[tgt], te["escalation_probability"]))
    out["baseline_vs_final"].append(dict(KPI=k["name"], Model="Majority-class baseline", **base_r))
    out["baseline_vs_final"].append(dict(KPI=k["name"], Model="Final Logistic Regression", **fin_r))
    out["errors"].setdefault(key, {})
    out["score_only_auc"] = out.get("score_only_auc", {})
    out["score_only_auc"][key] = score_auc

    tn, fp, fn, tp = confusion_matrix(te[tgt], pred).ravel()
    out["confusion"][key] = dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    out["flag_rate"][key] = float(pred.mean())
    out["base_rate"][key] = float(te[tgt].mean())

    # ---- 4. Error analysis by segment (evaluation only) ----
    t = te.assign(pred=pred, error=(pred != te[tgt].values).astype(int))
    segs = ["predicted_urgency", "submission_channel"] + (["recommended_department_id"] if key == "routing" else [])
    for s in segs:
        g = (t.groupby(s).agg(Cases=("error", "size"), Actual_Rate=(tgt, "mean"),
                              Error_Rate=("error", "mean")).reset_index().rename(columns={s: "Group"}))
        g["Actual_Rate"] = (g["Actual_Rate"] * 100).round(1)
        g["Error_Rate"] = (g["Error_Rate"] * 100).round(1)
        out["errors"][key][s] = g.sort_values("Cases", ascending=False).to_dict("records")

    # ---- 5. Overall predictor importance (permutation, ROC-AUC drop, 2026 test) ----
    pi = permutation_importance(model, te[feat], te[tgt], scoring="roc_auc", n_repeats=10,
                                random_state=42)
    out["importance"][key] = sorted(
        [dict(Predictor=f, Importance=float(v)) for f, v in zip(feat, pi.importances_mean)],
        key=lambda d: -d["Importance"])

    # ---- 6. Artifacts for the app: means of transformed features for explanations ----
    pre = model.named_steps["preprocessor"]
    Xtr = pre.transform(fin[feat])
    Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else np.asarray(Xtr)
    artifact["models"][key] = model
    artifact["kpis"][key] = dict(feature_names=list(pre.get_feature_names_out()),
                                 train_mean=Xtr.mean(axis=0), features=feat, cat=cat,
                                 threshold=thr, target=tgt)

# ---- Input ranges and plausibility relationships (from 2023-2025 data) ----
hist = esc[esc.year <= 2025]
ranges = {c: [float(hist[c].min()), float(hist[c].max())]
          for c in ["current_workload", "available_capacity", "known_financial_impact", "prior_complaint_count"]}
slope, intercept = np.polyfit(hist["current_workload"], hist["available_capacity"], 1)
resid_sd = float(np.std(hist["available_capacity"] - (slope * hist["current_workload"] + intercept)))
artifact["ranges"] = ranges
artifact["workload_capacity"] = dict(slope=float(slope), intercept=float(intercept), resid_sd=resid_sd)
artifact["wc_pairs"] = hist[["current_workload", "available_capacity"]].to_numpy()
artifact["categories"] = {"issue": sorted(esc["issue"].dropna().unique()),
                          "submission_channel": sorted(esc["submission_channel"].dropna().unique()),
                          "predicted_urgency": ["Low", "Medium", "High"],
                          "recommended_department_id": sorted(route["recommended_department_id"].dropna().unique())}
artifact["departments"] = dict(zip(dept["department_id"], dept["department_name"]))

joblib.dump(artifact, "models/final_models.joblib")
out["workload_capacity"] = artifact["workload_capacity"]
out["ranges"] = ranges
json.dump(out, open("models/metrics.json", "w"), indent=1, default=float)
print("Saved models/final_models.joblib and models/metrics.json")
for r in out["baseline_vs_final"]:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print("score-only AUC (2026):", out["score_only_auc"])
print("workload~capacity:", artifact["workload_capacity"])
