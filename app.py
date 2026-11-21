"""BankAssist - Streamlit dashboard for the multi-agent banking support system.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from bankassist.agents import LABELS
from bankassist.db import SupportDB
from bankassist.evaluation import load_saved, run_evaluation
from bankassist.llm import provider_name
from bankassist.orchestrator import SupportOrchestrator

st.set_page_config(page_title="BankAssist Multi-Agent Support", page_icon="🏦", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem}
div[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;margin-right:6px}
.Positive{background:#dcfce7;color:#166534}.Negative{background:#fee2e2;color:#991b1b}.Query{background:#dbeafe;color:#1e40af}
.path{background:#f1f5f9;color:#334155}
</style>""", unsafe_allow_html=True)

SCENARIOS = {
    "Positive Feedback Handler": ["Thanks for sorting out my net banking login issue.", "Thanks for resolving my credit card issue.",
                                  "The branch staff were extremely helpful today, great service!"],
    "Negative Feedback Handler": ["My debit card replacement still hasn't arrived.", "I was charged twice for the same UPI payment.",
                                  "Thanks, but my refund has still not been credited."],
    "Query Handler": ["Could you check the status of ticket 650932?", "What's the update on ticket 784521?",
                      "What is the status of my ticket?"],
}


@st.cache_resource
def get_system():
    db = SupportDB()
    return db, SupportOrchestrator(db)


db, orch = get_system()

with st.sidebar:
    st.title("🏦 BankAssist")
    st.caption("Multi-agent GenAI customer support")
    st.info(f"**LLM:** {provider_name()}")
    customer = st.text_input("Customer name", "Priya Sharma")
    st.markdown("**Agent graph (LangGraph)**")
    st.graphviz_chart("""digraph{rankdir=TB;node[shape=box,style="rounded,filled",fillcolor="#eef2ff",fontname=Helvetica,fontsize=10];
      U[label="User message",fillcolor="#f1f5f9"];C[label="Classifier Agent\\n(LLM, JSON output)"];
      P[label="Feedback Handler\\nPositive",fillcolor="#dcfce7"];N[label="Feedback Handler\\nNegative",fillcolor="#fee2e2"];
      Q[label="Query Handler",fillcolor="#dbeafe"];D[label="SQLite\\nsupport_tickets",shape=cylinder,fillcolor="#fef9c3"];
      L[label="agent_logs",shape=cylinder,fillcolor="#fef9c3"];
      U->C;C->P[label=" positive"];C->N[label=" negative"];C->Q[label=" query"];N->D[label=" INSERT"];Q->D[label=" SELECT"];
      P->L;N->L;Q->L}""")
    if st.button("Reset demo database", use_container_width=True):
        db.reset()
        st.session_state["chat"] = []
        st.rerun()

st.header("Banking Customer Support AI Agent")
logs = db.logs_df()
tickets = db.tickets_df()
m = st.columns(5)
m[0].metric("Messages handled", len(logs))
m[1].metric("Agent success rate", f"{100 * logs['success'].mean():.1f}%" if len(logs) else "–")
m[2].metric("Tickets in DB", len(tickets))
m[3].metric("Open tickets", int((tickets["status"] != "Resolved").sum() - (tickets["status"] == "Closed").sum()))
m[4].metric("👍 user rating", f"{(logs['user_rating'] == 1).sum()} / {logs['user_rating'].notna().sum()}" if len(logs) else "–")

tabs = st.tabs(["💬 Support Chat", "🧪 Test Scenarios", "🎫 Ticket Database", "📜 Logs & Debugging", "📊 Evaluation"])


def badge(label: str) -> str:
    return f"<span class='badge {label.split()[0]}'>{label}</span>"


def render_result(s: dict, key: str):
    st.markdown(badge(s["label"]) + f"<span class='badge path'>{' → '.join(s['agent_path'])}</span>"
                f"<span class='badge path'>confidence {s['confidence']:.2f}</span>"
                + (f"<span class='badge path'>🗄️ {s['ticket_action']}</span>" if s.get("ticket_action") not in (None, "none") else ""),
                unsafe_allow_html=True)
    st.markdown(s["response"])
    with st.expander("Prompt trace"):
        for t in s.get("trace", []):
            st.markdown(f"**{t['agent']}**")
            st.code(t["prompt"], language="text", wrap_lines=True)
            st.caption(f"Raw output: {t['raw_output']}")
    if s.get("log_id"):
        c1, c2, c3, _ = st.columns([1, 1, 3, 5])
        if c1.button("👍", key=f"up{key}"):
            db.rate(s["log_id"], 1)
            st.toast("Thanks for the feedback!")
        if c2.button("👎", key=f"dn{key}"):
            db.rate(s["log_id"], -1)
            st.toast("Logged for the improvement loop")
        fix = c3.selectbox("Correct label?", ["—"] + LABELS, key=f"fx{key}", label_visibility="collapsed")
        if fix != "—":
            db.rate(s["log_id"], -1 if fix != s["label"] else 1, fix)


# ---------------------------------------------------------------- chat -----
with tabs[0]:
    for i, msg in enumerate(st.session_state.get("chat", [])):
        with st.chat_message("user"):
            st.markdown(f"**{msg['customer_name']}:** {msg['message']}")
        with st.chat_message("assistant", avatar="🏦"):
            render_result(msg, f"chat{i}")
    text = st.chat_input("Type a customer message, e.g. 'Could you check the status of ticket 650932?'")
    if text:
        result = orch.handle(text, customer)
        if result["label"] == "ERROR":
            st.error(f"LLM request failed: {result['reason']}")
        else:
            st.session_state.setdefault("chat", []).append(result)
            st.rerun()

# ----------------------------------------------------------- scenarios -----
with tabs[1]:
    st.markdown("Run predefined scenarios for each agent role and verify routing end-to-end.")
    cols = st.columns(3)
    for col, (agent, msgs) in zip(cols, SCENARIOS.items()):
        with col:
            st.subheader(agent)
            for j, msg in enumerate(msgs):
                if st.button(msg, key=f"sc{agent}{j}", use_container_width=True):
                    st.session_state["scenario"] = (agent, orch.handle(msg, customer))
    if st.button("▶️ Run all scenarios", type="primary"):
        res = []
        for agent, msgs in SCENARIOS.items():
            for msg in msgs:
                s = orch.handle(msg, customer)
                res.append({"expected agent": agent, "message": msg, "classification": s["label"],
                            "agent path": " → ".join(s["agent_path"]), "routed correctly": s["agent_path"][-1] == agent,
                            "response": s["response"]})
        st.session_state["scenario_table"] = pd.DataFrame(res)
    if "scenario" in st.session_state:
        agent, s = st.session_state["scenario"]
        st.markdown(f"**Message:** {s['message']}")
        render_result(s, "scenario")
    if "scenario_table" in st.session_state:
        t = st.session_state["scenario_table"]
        st.success(f"Routing success: {t['routed correctly'].mean() * 100:.0f}% ({t['routed correctly'].sum()}/{len(t)})")
        st.dataframe(t, use_container_width=True, hide_index=True)

# -------------------------------------------------------------- tickets ----
with tabs[2]:
    a, b = st.columns([3, 1])
    with a:
        st.dataframe(tickets, use_container_width=True, hide_index=True, height=420)
    with b:
        st.markdown("**Update ticket status** (simulates the support team)")
        if len(tickets):
            tid = st.selectbox("Ticket", tickets["ticket_id"])
            status = st.selectbox("New status", ["Unresolved", "In Progress", "Resolved", "Closed"])
            if st.button("Update", use_container_width=True):
                db.update_status(tid, status)
                st.rerun()
        st.plotly_chart(px.pie(tickets, names="status", hole=0.5, title="Tickets by status").update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0)),
                        use_container_width=True)

# ----------------------------------------------------------------- logs ----
with tabs[3]:
    if logs.empty:
        st.caption("No interactions logged yet.")
    else:
        a, b, c = st.columns(3)
        by_agent = logs.assign(agent=logs["agent_path"].str.split(" → ").str[-1]).groupby("agent").agg(
            calls=("id", "count"), success_rate=("success", "mean")).reset_index()
        by_agent["success_rate"] = (by_agent["success_rate"] * 100).round(1)
        a.markdown("**Success / failure by agent**")
        a.dataframe(by_agent, hide_index=True, use_container_width=True)
        b.plotly_chart(px.histogram(logs, x="classification", color="classification", title="Classification distribution")
                       .update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=40, b=0)), use_container_width=True)
        tick = logs[logs["ticket_id"].notna()][["ts", "ticket_id", "ticket_action"]]
        c.markdown("**Ticket actions**")
        c.dataframe(tick, hide_index=True, use_container_width=True, height=220)
        f = st.multiselect("Filter by classification", logs["classification"].unique().tolist())
        view = logs[logs["classification"].isin(f)] if f else logs
        st.dataframe(view.drop(columns=["prompt_trace"]), use_container_width=True, hide_index=True, height=300)
        sel = st.selectbox("Inspect prompt trace for log id", view["id"])
        trace = json.loads(view.loc[view["id"] == sel, "prompt_trace"].iloc[0] or "[]")
        for t in trace:
            st.markdown(f"**{t['agent']}**")
            st.code(t["prompt"], language="text", wrap_lines=True)
            st.caption(f"Raw output: {t['raw_output']}")
        fb = logs[logs["corrected_label"].notna()]
        st.markdown("**Improvement loop — user-corrected labels** (exported as few-shot examples for the classifier prompt)")
        st.dataframe(fb[["message", "classification", "corrected_label"]], hide_index=True, use_container_width=True)
        st.download_button("Download agent logs (CSV)", logs.to_csv(index=False), "agent_logs.csv")

# ------------------------------------------------------------ evaluation ---
with tabs[4]:
    st.markdown("48 labelled test messages (16 per class, including informal and indirect phrasings) run through the **full "
                "agent graph** on an isolated DB. Query responses are graded with LangChain **QAEvalChain** against DB ground "
                "truth (with negative controls); empathy & clarity are rated 1–5 by an LLM judge.")
    if st.button("▶️ Run evaluation", type="primary"):
        with st.spinner("Running test suite..."):
            st.session_state["eval"] = run_evaluation()
    elif "eval" not in st.session_state and load_saved():
        st.session_state["eval"] = load_saved()
        st.caption("Showing the results of the last evaluation run (data/eval_*.csv). Click the button to re-run.")
    if "eval" in st.session_state:
        df, s, cm = st.session_state["eval"]
        r = st.columns(6)
        r[0].metric("Classification acc.", f"{s['classification_accuracy_%']}%")
        r[1].metric("Macro F1", f"{s['macro_f1']:.3f}")
        r[2].metric("Routing success", f"{s['routing_success_%']}%")
        r[3].metric("QAEvalChain (status)", f"{s['qa_status_accuracy_%']}%")
        r[4].metric("Empathy (neg.)", f"{s['avg_empathy_negative']}/5")
        r[5].metric("Clarity", f"{s['avg_clarity']}/5")
        a, b = st.columns(2)
        a.plotly_chart(px.imshow(cm.values, x=[c.replace("pred: ", "") for c in cm.columns], y=[i.replace("true: ", "") for i in cm.index],
                                 text_auto=True, color_continuous_scale="Blues", title="Confusion matrix (rows = true)")
                       .update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0)), use_container_width=True)
        b.markdown("**Per-class metrics**")
        b.dataframe(pd.DataFrame(s["per_class"]).T.round(3), use_container_width=True)
        b.metric("Wrong status answers caught by QAEvalChain grader", s["grader_negative_controls_caught"])
        if s["misclassified"]:
            b.markdown("**Misclassified messages**")
            b.dataframe(pd.DataFrame(s["misclassified"]), hide_index=True, use_container_width=True)
        st.dataframe(df[["message", "expected", "predicted", "routed_ok", "format_ok", "qa_grade", "empathy", "clarity", "response"]],
                     use_container_width=True, hide_index=True, height=380)
