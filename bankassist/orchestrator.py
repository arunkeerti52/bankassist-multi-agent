"""LangGraph orchestration: Classifier -> (Positive | Negative) Feedback Handler | Query Handler."""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from .agents import ClassifierAgent, FeedbackHandlerAgent, QueryHandlerAgent
from .db import SupportDB
from .llm import get_llm

ROUTES = {"Positive Feedback": "positive_feedback", "Negative Feedback": "negative_feedback", "Query": "query_handler"}


class SupportState(TypedDict, total=False):
    message: str
    customer_name: str
    label: str
    confidence: float
    reason: str
    product: str
    response: str
    ticket_id: Optional[str]
    ticket_action: str
    agent_path: list
    trace: list
    success: bool


FORMAT_CHECKS = {
    "positive_feedback": lambda s: s["response"].startswith(f"Thank you for your kind words, {s['customer_name']}!"),
    "negative_feedback": lambda s: bool(re.search(r"A new ticket #\d{6} has been generated", s["response"])),
    "query_handler": lambda s: bool(re.search(r"#\d{6} is currently marked as: \w+", s["response"])) or "ticket number" in s["response"]
                                   or bool(re.search(r"couldn't find ticket #\d{6}", s["response"])),
}


class SupportOrchestrator:
    def __init__(self, db: SupportDB | None = None):
        self.db = db or SupportDB()
        llm = get_llm()
        self.classifier = ClassifierAgent(llm)
        self.feedback = FeedbackHandlerAgent(self.db, llm)
        self.query = QueryHandlerAgent(self.db)
        self.graph = self._build()

    # ---- graph nodes ---------------------------------------------------------
    def _classify(self, s: SupportState) -> SupportState:
        r = self.classifier.run(s["message"])
        return {"label": r["label"], "confidence": r["confidence"], "reason": r["reason"], "product": r["product"],
                "agent_path": ["Classifier"], "trace": [r["trace"]]}

    def _positive(self, s: SupportState) -> SupportState:
        r = self.feedback.positive(s["message"], s["customer_name"])
        return {**self._merge(s, r, "Positive Feedback Handler")}

    def _negative(self, s: SupportState) -> SupportState:
        r = self.feedback.negative(s["message"], s["customer_name"], s.get("product", "general"))
        return {**self._merge(s, r, "Negative Feedback Handler")}

    def _query(self, s: SupportState) -> SupportState:
        r = self.query.run(s["message"], s["customer_name"])
        return {**self._merge(s, r, "Query Handler")}

    @staticmethod
    def _merge(s, r, agent):
        return {"response": r["response"], "ticket_id": r["ticket_id"], "ticket_action": r["ticket_action"],
                "agent_path": s["agent_path"] + [agent], "trace": s["trace"] + [r["trace"]]}

    def _build(self):
        g = StateGraph(SupportState)
        g.add_node("classifier", self._classify)
        g.add_node("positive_feedback", self._positive)
        g.add_node("negative_feedback", self._negative)
        g.add_node("query_handler", self._query)
        g.set_entry_point("classifier")
        g.add_conditional_edges("classifier", lambda s: ROUTES[s["label"]], list(ROUTES.values()))
        for n in ROUTES.values():
            g.add_edge(n, END)
        return g.compile()

    # ---- public API ------------------------------------------------------------
    def handle(self, message: str, customer_name: str = "Valued Customer", log: bool = True) -> dict:
        t0 = time.time()
        try:
            state = self.graph.invoke({"message": message, "customer_name": customer_name.strip() or "Valued Customer"})
            state["success"] = FORMAT_CHECKS[ROUTES[state["label"]]](state)
        except Exception as e:  # never crash the UI - log the failure
            state = {"message": message, "customer_name": customer_name, "label": "ERROR", "confidence": 0, "reason": str(e),
                     "response": "Sorry, something went wrong. A human agent will contact you.", "ticket_id": None,
                     "ticket_action": "none", "agent_path": ["Classifier", "ERROR"], "trace": [], "success": False}
        state["latency_ms"] = round((time.time() - t0) * 1000, 1)
        if log:
            state["log_id"] = self.db.log({
                "ts": datetime.now().isoformat(timespec="seconds"), "customer_name": customer_name, "message": message,
                "classification": state["label"], "confidence": state["confidence"],
                "agent_path": " → ".join(state["agent_path"]), "response": state["response"], "ticket_id": state.get("ticket_id"),
                "ticket_action": state.get("ticket_action"), "success": int(state["success"]),
                "latency_ms": state["latency_ms"], "prompt_trace": state.get("trace", [])})
        return state

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
