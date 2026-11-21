"""The three agents: Classifier, Feedback Handler, Query Handler."""
from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .db import SupportDB
from .llm import get_llm

LABELS = ["Positive Feedback", "Negative Feedback", "Query"]

CLASSIFIER_PROMPT = PromptTemplate.from_template("""\
You are the Classifier Agent of a bank's customer-support system.
Classify the customer message into exactly one category:
- "Positive Feedback": praise, gratitude or satisfaction (e.g. "Thanks for resolving my credit card issue.")
- "Negative Feedback": complaint, problem report, dissatisfaction, something not working/arrived (e.g. "My debit card replacement still hasn't arrived.")
- "Query": a request for information or the status of an existing ticket (e.g. "Could you check the status of ticket 650932?")
If a message thanks the bank but then reports an unresolved problem, it is Negative Feedback.
Also name the banking product/area involved (e.g. "debit card", "credit card", "net banking", "UPI payment", "loan",
"mobile app", "account", "ATM", "cheque", "refund", "fund transfer"; use "general" if none).
Respond with JSON only: {{"label": "<category>", "confidence": <0-1>, "reason": "<short reason>", "product": "<area>"}}

CUSTOMER MESSAGE: {message}
JSON:""")

THANK_YOU_PROMPT = PromptTemplate.from_template("""\
You are the Feedback Handler Agent of a bank. Write a warm, personalised thank-you reply (max 2 sentences).
It MUST start exactly with: "Thank you for your kind words, {name}! We're delighted to assist you."
Then add one short sentence that references what the customer praised. No promises, no new information.
CUSTOMER NAME: {name}
CUSTOMER MESSAGE: {message}
REPLY:""")

EMPATHY_PROMPT = PromptTemplate.from_template("""\
You are the Feedback Handler Agent of a bank. Write ONE short empathetic sentence (max 25 words) acknowledging
the specific problem below. Do not mention ticket numbers, timelines or compensation.
CUSTOMER NAME: {name}
CUSTOMER MESSAGE: {message}
SENTENCE:""")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


class ClassifierAgent:
    name = "Classifier"

    def __init__(self, llm=None):
        self.chain = CLASSIFIER_PROMPT | (llm or get_llm(0)) | StrOutputParser()

    def run(self, message: str, attempts: int = 2) -> dict:
        prompt = CLASSIFIER_PROMPT.format(message=message)
        raws = []
        for _ in range(attempts):  # retry once if the model returns malformed JSON or an unknown label
            raw = self.chain.invoke({"message": message})
            raws.append(raw)
            out = _parse_json(raw)
            label = next((l for l in LABELS if l.lower() == str(out.get("label", "")).lower()), None)
            if label:
                return {"label": label, "confidence": float(out.get("confidence", 0.5)), "reason": out.get("reason", ""),
                        "product": out.get("product") or "general",
                        "trace": {"agent": self.name, "prompt": prompt, "raw_output": raw}}
        raise ValueError(f"Classifier returned no valid label: {raws[-1][:200]}")


class FeedbackHandlerAgent:
    name = "Feedback Handler"

    def __init__(self, db: SupportDB, llm=None):
        llm = llm or get_llm(0.4)
        self.db = db
        self.thanks_chain = THANK_YOU_PROMPT | llm | StrOutputParser()
        self.empathy_chain = EMPATHY_PROMPT | llm | StrOutputParser()

    def positive(self, message: str, name: str) -> dict:
        reply = self.thanks_chain.invoke({"name": name, "message": message}).strip()
        required = f"Thank you for your kind words, {name}!"
        if not reply.startswith(required):  # enforce mandated format
            reply = f"{required} We're delighted to assist you. " + reply
        return {"response": reply, "ticket_id": None, "ticket_action": "none",
                "trace": {"agent": f"{self.name} (positive)", "prompt": THANK_YOU_PROMPT.format(name=name, message=message), "raw_output": reply}}

    def negative(self, message: str, name: str, category: str = "general") -> dict:
        ticket_id = self.db.create_ticket(name, message, category)
        empathy = self.empathy_chain.invoke({"name": name, "message": message}).strip().split("\n")[0]
        mandated = f"We apologize for the inconvenience. A new ticket #{ticket_id} has been generated, and our team will follow up shortly."
        return {"response": f"{empathy} {mandated}".strip(), "ticket_id": ticket_id,
                "ticket_action": f"INSERT support_tickets (#{ticket_id}, '{category}', status='Unresolved')",
                "trace": {"agent": f"{self.name} (negative)", "prompt": EMPATHY_PROMPT.format(name=name, message=message), "raw_output": empathy}}


class QueryHandlerAgent:
    name = "Query Handler"

    def __init__(self, db: SupportDB):
        self.db = db

    @staticmethod
    def extract_ticket(message: str) -> str | None:
        m = re.search(r"(?<!\d)(\d{6})(?!\d)", message)
        return m.group(1) if m else None

    def run(self, message: str, name: str) -> dict:
        tid = self.extract_ticket(message)
        trace = {"agent": self.name, "prompt": f"regex (?<!\\d)(\\d{{6}})(?!\\d) on: {message}", "raw_output": f"ticket={tid}"}
        if not tid:
            open_t = self.db.tickets_for(name)
            if len(open_t):
                listing = "; ".join(f"#{r.ticket_id} ({r.category}): {r.status}" for r in open_t.head(3).itertuples())
                resp = f"I couldn't find a ticket number in your message. Here are your recent tickets - {listing}. Share a 6-digit ticket number for details."
            else:
                resp = "Could you please share your 6-digit ticket number? I'll check its current status right away."
            return {"response": resp, "ticket_id": None, "ticket_action": "SELECT by customer (no ticket number)", "trace": trace, "found": False}
        ticket = self.db.get_ticket(tid)
        if not ticket:
            return {"response": f"I couldn't find ticket #{tid} in our system. Please double-check the number.",
                    "ticket_id": tid, "ticket_action": f"SELECT #{tid} -> not found", "trace": trace, "found": False}
        return {"response": f"Your ticket #{tid} is currently marked as: {ticket['status']}.", "ticket_id": tid,
                "ticket_action": f"SELECT #{tid} -> {ticket['status']}", "trace": trace, "found": True}
