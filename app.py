"""Customer Complaint Prioritization and Resolution Decision-Support Tool  (Week 11 version)

Run:  streamlit run app.py      (after:  python train.py)
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
from app_utils import (load_artifact, predict, predict_many, explain, validate, expected_capacity,
                       build_summary, changed_inputs, LABELS, KPI_NAMES, ADJUSTABLE)
from pipeline import build_queue

DATA_DIR = os.environ.get("COMPLAINT_DATA_DIR", "Data")
st.set_page_config(page_title="Complaint Decision Support", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifact(), json.load(open("models/metrics.json"))


@st.cache_data
def get_queue():
    art = load_artifact()
    q = build_queue(DATA_DIR, year=2026)
    for k in KPI_NAMES:
        q[f"{k}_risk"] = predict_many(art, k, q)
    return q


if not (os.path.exists("models/final_models.joblib") and os.path.exists("models/metrics.json")):
    st.error("Model files not found. Run `python train.py` first (see README).")
    st.stop()
try:
    art, M = get_artifacts()
    queue = get_queue()
except FileNotFoundError:
    st.error(f"Data files not found in '{DATA_DIR}/'. See README for how to obtain them.")
    st.stop()

THR = {k: art["kpis"][k]["threshold"] for k in KPI_NAMES}
DEPT = art["departments"]
pct = lambda x: f"{x*100:.1f}%"


def row_dict(r):
    d = {c: r[c] for c in ["prior_complaint_count", "known_financial_impact", "current_workload",
                           "available_capacity", "escalation_probability", "issue",
                           "submission_channel", "predicted_urgency", "recommended_department_id"]}
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def flag_text(kpi, p):
    return "Flag for review" if p >= THR[kpi] else "Below review threshold"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About this tool")
    st.write("This tool estimates two risks for each complaint to help the Customer Complaints Manager "
             "decide where to look first. It supports your judgment and never decides on its own.")
    st.markdown(f"**Review thresholds**  \nEscalation Risk: {THR['escalation']*100:.0f}%  \n"
                f"Routing Change Risk: {THR['routing']*100:.0f}%")
    st.caption("Complaints shown are 2026 complaints. Predictions come from a model trained only on "
               "2023–2025 complaints.")

st.title("Customer Complaint Decision-Support Tool")
tab1, tab2, tab3, tab4 = st.tabs(["1. Review a complaint", "2. What-if comparison",
                                  "3. How reliable is this tool?", "4. Data and method"])

# ------------------------------------------------------------------ complaint selection (shared)
with tab1:
    c1, c2, c3 = st.columns(3)
    urg = c1.multiselect("Predicted urgency", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    only_flag = c2.checkbox("Escalation flag only")
    top_n = c3.slider("Complaints listed", 10, 200, 50, 10)
    view = queue[queue["predicted_urgency"].isin(urg)]
    if only_flag:
        view = view[view["escalation_risk"] >= THR["escalation"]]
    view = view.sort_values("escalation_risk", ascending=False).head(top_n)
    if view.empty:
        st.info("No complaints match these filters.")
        st.stop()
    labels = {int(r.complaint_id): f"{int(r.complaint_id)} | {r.issue[:40]} | escalation {pct(r.escalation_risk)}"
              for r in view.itertuples()}
    cid = st.selectbox("Select a complaint (sorted by Escalation Risk)", list(labels), format_func=labels.get)
    st.session_state["cid"] = cid

cid = st.session_state["cid"]
rec = queue[queue["complaint_id"] == cid].iloc[0]
base = row_dict(rec)
base_pred = {k: predict(art, k, base) for k in KPI_NAMES}

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.subheader(f"Complaint {cid}")
    f1, f2 = st.columns(2)
    f1.markdown(f"**Product:** {rec['product']}  \n**Issue:** {rec['issue']}  \n**Sub-issue:** {rec['sub_issue']}  \n"
                f"**Channel:** {rec['submission_channel']}  \n**Received:** {rec['date_received']:%Y-%m-%d}")
    fin = base["known_financial_impact"]
    f2.markdown(f"**Predicted urgency:** {base['predicted_urgency']}  \n"
                f"**Earlier escalation score:** {pct(base['escalation_probability'])}  \n"
                f"**Recommended department:** {base['recommended_department_id']} – {DEPT.get(base['recommended_department_id'], '')}  \n"
                f"**Known financial impact:** {'not recorded' if pd.isna(fin) else f'${fin:,.2f}'}  \n"
                f"**Prior complaints:** {base['prior_complaint_count']}  \n"
                f"**Workload / capacity:** {base['current_workload']:g} / {base['available_capacity']:g}")
    m1, m2 = st.columns(2)
    for col, k in zip((m1, m2), KPI_NAMES):
        col.metric(KPI_NAMES[k], pct(base_pred[k]), flag_text(k, base_pred[k]),
                   delta_color="off" if base_pred[k] < THR[k] else "inverse")
    st.caption("A flag means “worth a closer look”, not “this will happen”. See tab 3 for how often flags are right.")

    st.markdown("#### Why this prediction? (this complaint only)")
    st.caption("Each factor shows how far it moves THIS complaint's risk above or below a typical complaint. "
               "This explains how the model reached its number. It does not prove that changing a factor "
               "would change the real outcome.")
    e1, e2 = st.columns(2)
    for col, k in zip((e1, e2), KPI_NAMES):
        ex = explain(art, k, base)
        col.markdown(f"**{KPI_NAMES[k]}**")
        col.dataframe(ex, hide_index=True, width="stretch")

    with st.expander("Overall predictor importance (all complaints, not this one)"):
        st.caption("This is different from the table above: it shows which factors matter most across ALL "
                   "2026 complaints (drop in ranking accuracy when a factor is scrambled).")
        for k in KPI_NAMES:
            imp = pd.DataFrame(M["importance"][k])
            imp["Predictor"] = imp["Predictor"].map(lambda c: LABELS.get(c, c))
            st.markdown(f"**{KPI_NAMES[k]}**")
            st.bar_chart(imp.set_index("Predictor")["Importance"].clip(lower=0))
        st.caption("The earlier escalation score dominates both models. Workload and capacity have almost no influence.")

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader(f"What-if comparison for complaint {cid}")
    st.write("Change only the things a manager could realistically influence, then compare against the baseline.")
    with st.expander("Which inputs can I change?", expanded=True):
        st.markdown(
            "**You can change (decision levers / conditions):** current workload, available capacity, "
            "recommended department.  \n"
            "**Fixed (facts about the complaint, shown but locked):** issue, submission channel, prior complaints, "
            "known financial impact.  \n"
            "**Locked (outputs of the earlier scoring stage):** predicted urgency and the earlier escalation score. "
            "They are model results, not something a manager can change, so editing them would be misleading.")
    nonce = st.session_state.setdefault("nonce", 0)
    scen = {}
    cols = st.columns(3)
    cols[0].markdown("**Baseline** (current values)")
    cols[0].write(f"Workload: **{base['current_workload']:g}**")
    cols[0].write(f"Capacity: **{base['available_capacity']:g}**")
    cols[0].write(f"Department: **{base['recommended_department_id']}**")
    for col, name in zip(cols[1:], ["Scenario A", "Scenario B"]):
        col.markdown(f"**{name}**")
        k = f"{name}_{cid}_{nonce}"
        wl = col.number_input("Current workload", 0.0, 40.0, float(base["current_workload"]), 0.1, key=f"wl_{k}")
        cap = col.number_input("Available capacity", 0, 300, int(base["available_capacity"]), 1, key=f"cap_{k}")
        depts = art["categories"]["recommended_department_id"]
        dep = col.selectbox("Recommended department", depts, index=depts.index(base["recommended_department_id"]),
                            format_func=lambda d: f"{d} – {DEPT.get(d, '')}", key=f"dep_{k}")
        band = expected_capacity(art, wl)
        if band:
            col.caption(f"At workload ≈{wl:g}, past capacity ranged {band['lo']:g}–{band['hi']:g}.")
        scen[name] = {**base, "current_workload": float(wl), "available_capacity": int(cap),
                      "recommended_department_id": dep}
    if st.button("Reset scenarios"):
        st.session_state["nonce"] += 1
        st.rerun()

    results = {}
    for name, row in scen.items():
        errs, warns = validate(art, row, base)
        for e in errs:
            st.error(f"{name}: {e}")
        for w in warns:
            st.warning(f"{name}: {w}")
        results[name] = (row, None if errs else {k: predict(art, k, row) for k in KPI_NAMES})

    st.markdown("#### Side-by-side comparison")
    table = {"Baseline": {}, "Scenario A": {}, "Scenario B": {}}
    for lab, c, f in [("Current workload", "current_workload", lambda v: f"{v:g}"),
                      ("Available capacity", "available_capacity", lambda v: f"{v:g}"),
                      ("Recommended department", "recommended_department_id", str)]:
        table["Baseline"][lab] = f(base[c])
        for n in scen:
            v = scen[n][c]
            table[n][lab] = f(v) + (" ◄ changed" if v != base[c] else "")
    for k, nm in KPI_NAMES.items():
        table["Baseline"][f"{nm}"] = pct(base_pred[k])
        table["Baseline"][f"{nm}: change vs baseline"] = "–"
        table["Baseline"][f"{nm}: review decision"] = flag_text(k, base_pred[k])
        for n, (row, pr) in results.items():
            if pr is None:
                table[n][nm] = table[n][f"{nm}: change vs baseline"] = table[n][f"{nm}: review decision"] = "— (invalid input)"
            else:
                table[n][nm] = pct(pr[k])
                table[n][f"{nm}: change vs baseline"] = f"{(pr[k]-base_pred[k])*100:+.1f} pct. points"
                table[n][f"{nm}: review decision"] = flag_text(k, pr[k])
    st.table(pd.DataFrame(table))

    st.markdown("#### Automatic summary")
    st.info(build_summary(art, base, results, base_pred))

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("How much can you rely on these predictions?")
    cf, bv = M["confusion"], {(r["KPI"], r["Model"]): r for r in M["baseline_vs_final"]}
    e, r_ = cf["escalation"], cf["routing"]
    e_rec = e["TP"] / (e["TP"] + e["FN"])
    st.markdown(
        f"**In short:** the tool is better than guessing, but only moderately. Use it to decide *where to look first*, "
        f"not to decide *what is true*.\n\n"
        f"- **Escalation Risk:** of every 100 complaints that really escalated, the tool flagged about "
        f"**{e_rec*100:.0f}**. But it flagged about **{M['flag_rate']['escalation']*100:.0f} of every 100** complaints, "
        f"and only about half of flagged complaints actually escalated (vs {M['base_rate']['escalation']*100:.0f}% of all "
        f"complaints). A flag is therefore a weak signal.\n"
        f"- **Routing Change Risk:** about **{bv[('Routing Change Risk','Final Logistic Regression')]['Recall']*100:.0f}%** of real "
        f"routing changes were flagged, and about **{bv[('Routing Change Risk','Final Logistic Regression')]['Precision']*100:.0f}%** "
        f"of flagged complaints really changed department.\n"
        f"- The model works best for High-urgency complaints and is least reliable for **Medium-urgency** complaints "
        f"and for complaints recommended to department **D02**.")
    st.markdown("#### Final test results (2026 complaints, tested once)")
    bf = pd.DataFrame(M["baseline_vs_final"])
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        bf[c] = bf[c].map(lambda v: f"{v:.3f}")
    st.dataframe(bf.rename(columns={"ROC_AUC": "ROC-AUC"}), hide_index=True, width="stretch")
    st.caption("Baseline = always predict the most common outcome. ROC-AUC 0.50 means no better than chance; "
               "1.00 would be perfect.")
    sc = M["score_only_auc"]
    st.markdown(f"**Important:** for Escalation Risk, the earlier escalation score alone already reaches a ROC-AUC of "
                f"{sc['escalation']:.3f}; the full model reaches "
                f"{bv[('Escalation Risk','Final Logistic Regression')]['ROC_AUC']:.3f}. The extra variables add very little.")

    st.markdown("#### Where the tool makes more mistakes (2026)")
    for k, segs in M["errors"].items():
        for s, rows in segs.items():
            st.markdown(f"**{KPI_NAMES[k]} by {LABELS.get(s, s).lower()}**")
            st.dataframe(pd.DataFrame(rows).rename(columns={"Actual_Rate": "Actual outcome rate (%)", "Error_Rate": "Wrong predictions (%)"}),
                         hide_index=True, width="stretch")
    st.caption("Groups with few complaints (for example Postal mail or Referral) give unstable percentages.")

    st.markdown("#### Did we improve the models? (2025 development data)")
    dc = pd.DataFrame(M["dev_comparison"])
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        dc[c] = dc[c].map(lambda v: f"{v:.3f}")
    st.dataframe(dc.rename(columns={"ROC_AUC": "ROC-AUC"}), hide_index=True, width="stretch")
    st.caption(f"We tried balancing the classes and a Random Forest. Neither improved ROC-AUC, so we kept the original "
               f"Logistic Regression ({M['retained_model']}). The 2026 data was not used to choose.")

    st.markdown("#### Limitations")
    st.markdown("- Accuracy is moderate (ROC-AUC about 0.64–0.66); many flags are false alarms.\n"
                "- Some operational and outcome data are synthetic, so real-world performance may differ.\n"
                "- Escalation Risk depends mainly on an earlier model's score.\n"
                "- What-if results show how the model's numbers change, not what would happen in reality.\n"
                "- Results are single estimates with no confidence intervals; small differences may be noise.")

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("How the data was split")
    sp = M["split"]
    st.table(pd.DataFrame([
        {"Stage": "Training (for model comparison)", "Years": "2023–2024", "Escalation rows": sp["escalation"]["train_2023_2024"], "Routing rows": sp["routing"]["train_2023_2024"]},
        {"Stage": "Development (compare and choose models)", "Years": "2025", "Escalation rows": sp["escalation"]["dev_2025"], "Routing rows": sp["routing"]["dev_2025"]},
        {"Stage": "Final training (retained model)", "Years": "2023–2025", "Escalation rows": sp["escalation"]["final_train_2023_2025"], "Routing rows": sp["routing"]["final_train_2023_2025"]},
        {"Stage": "Final test (used once)", "Years": "2026", "Escalation rows": sp["escalation"]["test_2026"], "Routing rows": sp["routing"]["test_2026"]},
    ]))
    st.write("The split is by time because the tool uses past complaints to help with future ones. "
             "The 2026 data was never used to tune or select models or thresholds.")
    st.markdown("#### Input checks used in the what-if tab")
    wc = M["ranges"]
    st.markdown(f"- Workload must be within the range seen in past data ({wc['current_workload'][0]:g}–{wc['current_workload'][1]:g}); capacity within {wc['available_capacity'][0]:g}–{wc['available_capacity'][1]:g}, whole numbers only.\n"
                "- Workload and capacity are linked in the data (higher workload goes with lower capacity). A combination that has never occurred is blocked; an unusual one shows a warning.\n"
                "- Only departments the tool can recommend are allowed.\n"
                "- Invalid scenarios show a message and no prediction.")
