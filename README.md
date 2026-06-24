# AgentMira — Buyer Lead Intake Agent

> Engineering case study submission. Processes incoming real estate buyer inquiries and produces actionable **Lead Briefs** for realtors using an agentic LLM reasoning loop.

---

## How it works

```
Buyer inquiry (free text)
        │
        ▼
┌───────────────┐
│  Guard Layer  │  Injection detection · impossible budgets · identity checks
└──────┬────────┘
       │ sanitised message + flags
       ▼
┌─────────────────────────────────────────────────┐
│           Agentic Reasoning Loop                │
│                                                 │
│  LLM reasons → picks a tool → reads result     │
│       └──────────────────────────┘             │
│            repeats up to 8 turns               │
│                                                 │
│  Tools:                                         │
│  • search_listings      filter + rank MLS       │
│  • get_listing_detail   full record by ID       │
│  • neighborhood_stats   price benchmarks        │
└──────────────────┬──────────────────────────────┘
                   │ structured JSON findings + tool trace
                   ▼
         ┌──────────────────┐
         │  Brief Generator │  Second LLM call → Markdown brief
         └────────┬─────────┘
                  │
       ┌──────────┴──────────┐
  Lead Brief (.md)    Realtor Alerts + Confidence Score
```

The LLM does not receive a pre-filtered list. It decides what to search, adapts based on results, and loops until it has enough context. That reasoning trace is logged and visible.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/agentmira-submission
cd agentmira-submission
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=your-key-here
# or create a .env file: ANTHROPIC_API_KEY=your-key-here

# 3. Run all 12 leads
python main.py

# 4. Run a single lead
python main.py --lead LEAD-2026-001

# 5. Run multiple specific leads
python main.py --lead LEAD-2026-001 LEAD-2026-006 LEAD-2026-010
```

---

## Output

| File | Contents |
|------|----------|
| `output/briefs/LEAD-XXXX-XXX.md` | Lead Brief the realtor reads |
| `output/briefs/LEAD-XXXX-XXX.json` | Full result including tool trace |
| `output/all_briefs.md` | All 12 briefs in one file |
| `logs/run_TIMESTAMP.json` | Complete run log |

---

## Project structure

```
agentmira-submission/
├── main.py               CLI entry point
├── requirements.txt
├── src/
│   ├── guard.py          Pre-LLM safety and anomaly detection
│   ├── tools.py          MLS tool functions + Anthropic tool definitions
│   └── agent.py          Agentic loop + brief writer
└── data/
    ├── miami_mls_listings.csv
    └── sample_buyer_inquiries.json
```

---

## The three tools

| Tool | What it does | When the agent calls it |
|------|-------------|------------------------|
| `search_listings` | Filter + score + rank listings by neighbourhood, price, beds, type, features | Primary discovery — called 1–3× per lead with evolving params |
| `get_listing_detail` | Full MLS record for a specific ID | After identifying top matches, or when buyer mentions an address |
| `neighborhood_stats` | Median price, range, inventory for a neighbourhood | When budget seems misaligned with preferred area |

---

## Edge cases handled

| Lead | Problem | How the agent handles it |
|------|---------|--------------------------|
| LEAD-2026-003 | Anonymous buyer, $250K for 4BR + ocean view + pool in Brickell | Guard flags impossible budget; agent confirms via `neighborhood_stats`; brief explains gap |
| LEAD-2026-004 | Near-empty message, zero criteria | Guard flags thin message; brief is entirely qualifying questions |
| LEAD-2026-005 | Asking for negotiation advice on a specific listing | Agent retrieves listing, notes 50 sqft data anomaly; brief redirects to realtor judgment |
| LEAD-2026-006 | Prompt injection attack embedded in message | Guard strips injected content before LLM sees it; processes legitimate request |
| LEAD-2026-008 | 180-word rambling message, real needs buried | Agent extracts structured intent; brief surfaces clean specs |
| LEAD-2026-009 | Cash buyer | Flagged as high priority in brief |
| LEAD-2026-012 | Investor wanting 2–3 properties | Agent runs separate multi-family + condo searches; portfolio shortlist |

---

## Design decisions

**One agent, not multiple.** Each lead is a single synchronous task with no meaningful parallelism. A tool-calling loop gives visible intermediate reasoning without inter-agent communication overhead.

**Guard layer is regex, not AI.** Injection detection runs before any LLM call — fast, auditable, zero token cost.

**Tools are pandas, not another LLM.** The MLS dataset is small and structured. Deterministic filtering is faster and more predictable. The intelligence lives in which queries the agent chooses to run.

**Never block a lead, always flag.** Even impossible budgets and injection attempts produce a Lead Brief. Realtors need to know about anomalies.

**Owner PII never reaches the LLM.** Names and phone numbers are stripped from tool output before entering the context window.

---

## Model

`claude-sonnet-4-6` — change `MODEL` in `src/agent.py` to switch.
