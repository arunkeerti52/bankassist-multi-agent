"""LLMOps evaluation: classification test coverage, routing success, QA scoring and response quality."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import classification_report, confusion_matrix

try:
    from langchain_classic.evaluation.qa import QAEvalChain
except ImportError:
    from langchain.evaluation.qa import QAEvalChain

from .agents import LABELS
from .db import SupportDB
from .llm import get_llm
from .orchestrator import SupportOrchestrator

ROOT = Path(__file__).resolve().parent.parent
EVAL_DB = ROOT / "data" / "eval_support.db"
RESULTS = ROOT / "data" / "eval_results.csv"

TEST_CASES = [  # 48 labelled messages, 16 per class
    # ---- Positive feedback -------------------------------------------------
    ("Thanks for resolving my credit card issue.", "Positive Feedback"),
    ("Thanks for sorting out my net banking login issue.", "Positive Feedback"),
    ("The branch staff were extremely helpful today, great service!", "Positive Feedback"),
    ("I appreciate how quickly my loan was approved.", "Positive Feedback"),
    ("Your mobile app is smooth and easy to use, love it.", "Positive Feedback"),
    ("Kudos to the support team for fixing my UPI problem so fast.", "Positive Feedback"),
    ("Very happy with the new fixed deposit rates.", "Positive Feedback"),
    ("Excellent experience opening my savings account online.", "Positive Feedback"),
    ("My refund was credited within a day - impressed!", "Positive Feedback"),
    ("Thank you, the agent explained the EMI schedule perfectly.", "Positive Feedback"),
    ("Wonderful service at the ATM helpdesk, well done.", "Positive Feedback"),
    ("Grateful for the prompt help with my blocked card.", "Positive Feedback"),
    # ---- Negative feedback -------------------------------------------------
    ("My debit card replacement still hasn't arrived.", "Negative Feedback"),
    ("I was charged twice for the same transaction.", "Negative Feedback"),
    ("The mobile app keeps crashing when I try to transfer funds.", "Negative Feedback"),
    ("Money was deducted but the UPI payment failed.", "Negative Feedback"),
    ("Your customer care was rude and unhelpful.", "Negative Feedback"),
    ("I've been waiting two weeks for my cheque book.", "Negative Feedback"),
    ("There is an unauthorized transaction on my credit card.", "Negative Feedback"),
    ("Net banking is not working since yesterday.", "Negative Feedback"),
    ("Thanks, but my refund has still not been credited.", "Negative Feedback"),
    ("Terrible experience, my account got locked for no reason.", "Negative Feedback"),
    ("The hidden charges on my statement are unacceptable.", "Negative Feedback"),
    ("ATM swallowed my card and nobody has called back.", "Negative Feedback"),
    # ---- Queries -----------------------------------------------------------
    ("Could you check the status of ticket 650932?", "Query"),
    ("What's the update on ticket 784521?", "Query"),
    ("Can you tell me the status of my complaint 312874?", "Query"),
    ("Is ticket 459013 resolved yet?", "Query"),
    ("Any progress on 901245?", "Query"),
    ("Please track ticket number 128456 for me.", "Query"),
    ("What is the status of ticket 567890?", "Query"),
    ("How long will ticket 784521 take?", "Query"),
    ("Check ticket 999999 please", "Query"),
    ("What is the status of my ticket?", "Query"),
    ("How do I update my KYC details?", "Query"),
    ("Can you tell me when my ticket will be closed?", "Query"),
    # ---- harder phrasings (informal praise, indirect complaints, terse queries) ----
    ("Shout-out to Ravi at the Andheri branch, super courteous.", "Positive Feedback"),
    ("The new credit card rewards are fantastic, thanks a lot!", "Positive Feedback"),
    ("Brilliant - my NEFT went through in seconds.", "Positive Feedback"),
    ("Nice work on the app update, transfers feel faster.", "Positive Feedback"),
    ("I got an OTP for a transaction I never made!", "Negative Feedback"),
    ("Your IVR keeps disconnecting me, this is ridiculous.", "Negative Feedback"),
    ("Interest for March is missing from my FD payout.", "Negative Feedback"),
    ("The loan officer never returned my calls.", "Negative Feedback"),
    ("status 650932", "Query"),
    ("Has anyone looked at 312874 yet?", "Query"),
    ("Where can I see the progress of ticket 784521?", "Query"),
    ("I want to know if 459013 is closed", "Query"),
]

JUDGE_PROMPT = PromptTemplate.from_template("""\
You are a QA reviewer for a bank's support bot. Rate the RESPONSE to the CUSTOMER MESSAGE on two 1-5 scales:
- empathy: warmth/acknowledgement of the customer's feelings appropriate to the situation
- clarity: is the outcome (thanks / ticket number / ticket status) clear and unambiguous
Return JSON only: {{"empathy": <1-5>, "clarity": <1-5>}}
CUSTOMER MESSAGE: {message}
RESPONSE: {response}
SCORES:""")


def run_evaluation() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    db = SupportDB(EVAL_DB)
    db.reset()  # fresh, deterministic DB for every run
    orch = SupportOrchestrator(db)
    llm = get_llm(0)
    judge = JUDGE_PROMPT | llm | StrOutputParser()
    expected_path = {"Positive Feedback": "Positive Feedback Handler", "Negative Feedback": "Negative Feedback Handler",
                     "Query": "Query Handler"}
    rows = []
    for msg, gold in TEST_CASES:
        s = orch.handle(msg, "Test Customer")
        scores = {}
        try:
            scores = json.loads(re.search(r"\{.*\}", judge.invoke({"message": msg, "response": s["response"]}), re.S).group(0))
        except Exception:
            pass
        ticket_ok = True
        if s["label"] == "Negative Feedback":
            ticket_ok = bool(s.get("ticket_id")) and db.ticket_exists(s["ticket_id"])
        rows.append({"message": msg, "expected": gold, "predicted": s["label"], "confidence": s["confidence"],
                     "correct": s["label"] == gold,
                     "routed_ok": s["agent_path"][-1] == expected_path[gold] and s["success"] and ticket_ok,
                     "format_ok": s["success"], "empathy": scores.get("empathy"), "clarity": scores.get("clarity"),
                     "response": s["response"], "ticket_id": s.get("ticket_id"), "latency_ms": s["latency_ms"]})
    df = pd.DataFrame(rows)

    # QAEvalChain on the Query-handler responses (reference built from the DB ground truth)
    q = df[(df["expected"] == "Query") & df["message"].str.contains(r"\d{6}")]
    examples, preds = [], []
    for r in q.itertuples():
        tid = re.search(r"\d{6}", r.message).group(0)
        t = db.get_ticket(tid)
        ref = f"Your ticket #{tid} is currently marked as: {t['status']}." if t else f"Ticket #{tid} could not be found."
        examples.append({"question": r.message, "answer": ref})
        preds.append({"result": r.response})
    graded = QAEvalChain.from_llm(llm).evaluate(examples, preds, question_key="question", answer_key="answer", prediction_key="result")
    qa_ok = ["INCORRECT" not in (g.get("results") or g.get("text", "")).upper() for g in graded]
    # Negative control: wrong status answers must be graded INCORRECT
    ctrl_ex = [{"question": "Could you check the status of ticket 650932?", "answer": "Your ticket #650932 is currently marked as: Resolved."},
               {"question": "What's the update on ticket 784521?", "answer": "Your ticket #784521 is currently marked as: In Progress."}]
    ctrl_pred = [{"result": "Your ticket #650932 is currently marked as: Unresolved."},
                 {"result": "Your ticket #123456 is currently marked as: Closed."}]
    ctrl = QAEvalChain.from_llm(llm).evaluate(ctrl_ex, ctrl_pred, question_key="question", answer_key="answer", prediction_key="result")
    caught = sum("INCORRECT" in (g.get("results") or g.get("text", "")).upper() for g in ctrl)
    df["qa_grade"] = None
    df.loc[q.index, "qa_grade"] = ["CORRECT" if ok else "INCORRECT" for ok in qa_ok]

    report = classification_report(df["expected"], df["predicted"], labels=LABELS, output_dict=True, zero_division=0)
    summary = {
        "test_cases": len(df),
        "classification_accuracy_%": float(round(100 * df["correct"].mean(), 1)),
        "macro_f1": float(round(report["macro avg"]["f1-score"], 3)),
        "routing_success_%": float(round(100 * df["routed_ok"].mean(), 1)),
        "format_compliance_%": float(round(100 * df["format_ok"].mean(), 1)),
        "qa_status_accuracy_%": round(100 * sum(qa_ok) / max(len(qa_ok), 1), 1),
        "avg_empathy_negative": float(round(df.loc[df.expected == "Negative Feedback", "empathy"].astype(float).mean(), 2)),
        "avg_empathy_positive": float(round(df.loc[df.expected == "Positive Feedback", "empathy"].astype(float).mean(), 2)),
        "avg_clarity": float(round(df["clarity"].astype(float).mean(), 2)),
        "avg_latency_ms": float(round(df["latency_ms"].mean(), 1)),
        "grader_negative_controls_caught": f"{caught}/{len(ctrl)}",
        "misclassified": df.loc[~df["correct"], ["message", "expected", "predicted"]].to_dict("records"),
        "per_class": {l: {k: float(round(v, 3)) for k, v in report[l].items()} for l in LABELS},
    }
    cm = pd.DataFrame(confusion_matrix(df["expected"], df["predicted"], labels=LABELS),
                      index=[f"true: {l}" for l in LABELS], columns=[f"pred: {l}" for l in LABELS])
    df.to_csv(RESULTS, index=False)
    (ROOT / "data" / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return df, summary, cm


def load_saved():
    """Results of the last evaluation run (so the dashboard can show them without spending LLM quota)."""
    summary_f = ROOT / "data" / "eval_summary.json"
    if not (RESULTS.exists() and summary_f.exists()):
        return None
    df, summary = pd.read_csv(RESULTS), json.loads(summary_f.read_text())
    cm = pd.DataFrame(confusion_matrix(df["expected"], df["predicted"], labels=LABELS),
                      index=[f"true: {l}" for l in LABELS], columns=[f"pred: {l}" for l in LABELS])
    return df, summary, cm


if __name__ == "__main__":
    d, s, cm = run_evaluation()
    print(d[~d.correct][["message", "expected", "predicted"]].to_string())
    print(cm)
    print(json.dumps({k: v for k, v in s.items() if k != "per_class"}, indent=2))
