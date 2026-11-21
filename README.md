# BankAssist: Banking Customer Support AI Agent (Multi-Agent)

A LangGraph multi-agent system: a Classifier Agent routes each message to a Positive or Negative Feedback Handler or a Query Handler. It is backed by a SQLite `support_tickets` table and includes evaluation, logging and a Streamlit dashboard.

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env        # set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
streamlit run app.py
python -m bankassist.evaluation
```

## Project layout
| Path | Purpose |
|---|---|
| `bankassist/agents.py` | Classifier, Feedback Handler and Query Handler agents (prompts + logic) |
| `bankassist/orchestrator.py` | LangGraph StateGraph, routing, format checks, logging |
| `bankassist/db.py` | SQLite `support_tickets` and `agent_logs`, feedback loop |
| `bankassist/evaluation.py` | 48-message suite: classification, routing, QAEvalChain (with negative controls), empathy/clarity judge |
| `bankassist/llm.py` | Cloudflare Workers AI client (Llama 3.3 70B) |
| `app.py` | Streamlit dashboard (chat, scenarios, tickets, logs, evaluation) |
