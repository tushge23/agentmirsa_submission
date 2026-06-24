"""
agent.py — Agentic reasoning loop + Lead Brief generator.

Stage 1 (this file, first LLM call):
  LLM receives sanitised buyer message + tool definitions.
  It calls tools iteratively (up to MAX_TOOL_TURNS) until it has
  enough context, then outputs structured JSON research findings.

Stage 2 (this file, second LLM call):
  A separate Brief Writer LLM prompt takes the research findings
  and writes the final Markdown Lead Brief the realtor reads.

Every tool call is logged to a trace for auditability.
"""

import json
import anthropic
from datetime import datetime, timezone

from . import guard
from . import tools as mls_tools

MODEL = "claude-sonnet-4-6"
MAX_TOOL_TURNS = 8


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_AGENT = """\
You are a real estate lead intake agent for a licensed Miami realtor.

Your job: analyse an incoming buyer inquiry, use MLS search tools to find the best \
matching properties, and return structured JSON research findings. A separate system \
will write the final brief — focus on RESEARCH and REASONING.

## Research process
1. Extract from the buyer message: budget, bedrooms, neighbourhood preferences, \
property type, must-have features, nice-to-have features, urgency, lead type \
(owner-occupier vs investor), and any special context.

2. Search for matching listings. Adapt your strategy based on what you find:
   - Start with the buyer's preferred neighbourhood(s).
   - If results are thin, call neighborhood_stats to understand why, then search \
adjacent or comparable areas.
   - If the budget seems misaligned with the preferred area, confirm with \
neighborhood_stats and explain in your findings.
   - For investors, search multi-family AND rentable condos separately.
   - Call search_listings multiple times with different parameters — this is expected.

3. For your top 2–3 matches, call get_listing_detail for the full record.

4. Stop when you have 3–5 solid matches OR a clear, honest explanation of why \
matches are unavailable.

## Confidence scoring
Assess your own confidence in the research findings:
- HIGH: Budget is clear, neighbourhood specified, multiple strong matches found.
- MEDIUM: Some criteria missing or budget is tight, but reasonable matches exist.
- LOW: Requirements are vague, budget is unrealistic, or almost no matches found.
- VERY LOW: Almost no usable criteria provided, or request is impossible to fulfil.

## Rules
- NEVER reveal owner names, phone numbers, or any personal contact information.
- Be honest about limitations — do not fabricate matches.
- Your reasoning will be logged and shown to the realtor.

## Output
When done, output ONLY a valid JSON object:
{
  "lead_type": "owner_occupier" | "investor" | "unclear",
  "confidence": "HIGH" | "MEDIUM" | "LOW" | "VERY LOW",
  "confidence_reasons": ["reason 1", "reason 2"],
  "extracted_intent": {
    "budget_min": null,
    "budget_max": null,
    "bedrooms_min": null,
    "bedrooms_max": null,
    "neighborhoods_preferred": [],
    "property_types": [],
    "must_have_features": [],
    "nice_to_have_features": [],
    "urgency": "immediate" | "within_months" | "flexible" | "unknown",
    "special_context": ""
  },
  "top_matches": [
    {
      "listing_id": "",
      "address": "",
      "price": 0,
      "bedrooms": 0,
      "sqft": 0,
      "neighborhood": "",
      "property_type": "",
      "why_this_match": "1–2 sentences explaining fit"
    }
  ],
  "search_summary": "What you searched, what you found, any budget mismatches or surprises",
  "agent_flags": ["additional concerns not already raised by the guard layer"]
}"""


SYSTEM_PROMPT_BRIEF_WRITER = """\
You are a senior real estate assistant writing a Lead Brief for a busy realtor.

The realtor reads this brief on their phone seconds before calling the buyer. \
Write it to be FAST to read and IMMEDIATELY actionable. No fluff, no filler.

## Format (Markdown)

### 📋 Lead Snapshot
One line: name | what they want | budget | urgency | lead type

### 🎯 What They Want
Bullet points extracted from their message. Be specific. Include any important \
personal context (relocating, elderly parents, investor, etc.).

### 🏠 Recommended Properties
For each property:
**[Address]** — $X | X bed / X bath | X sqft | [Neighbourhood]
Match: 🟢 Strong / 🟡 Good / 🟠 Partial
> Why this fits: [1–2 sentences]

### ⚠️ Realtor Alerts
All flags from the guard layer and agent, written plainly. \
If none, write "None."

### 🤖 Agent Confidence
State: HIGH / MEDIUM / LOW / VERY LOW
Reasons: [bullet list]

### ✅ Suggested First Move
One clear, specific action for the realtor.

## Rules
- Never reveal owner names or phone numbers.
- If no good matches exist, say so honestly — do not invent properties.
- Adjust tone for context: warmer for first-time buyers, direct for investors."""


# ── Agent loop ────────────────────────────────────────────────────────────────

def _run_agent_loop(
    lead: dict,
    guard_result: guard.GuardResult,
    client: anthropic.Anthropic,
) -> tuple[dict, list[dict]]:
    """
    Agentic reasoning loop. LLM calls tools iteratively until it has
    enough context, then outputs JSON findings.
    Returns (findings_dict, tool_trace_list).
    """
    trace: list[dict] = []

    user_message = f"""\
Buyer inquiry to process:

Lead ID:      {lead['lead_id']}
Received:     {lead['received_at']}
Channel:      {lead['channel']}
Buyer name:   {lead['buyer_name']}
Email:        {lead['buyer_email']}
Phone:        {lead.get('buyer_phone') or 'not provided'}

Message (sanitised):
{guard_result.sanitized_message}

Guard layer flags already raised:
{json.dumps(guard_result.flags, indent=2) if guard_result.flags else 'None'}

Research this lead using the available tools, then return your findings as JSON."""

    messages = [{"role": "user", "content": user_message}]

    for turn in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT_AGENT,
            tools=mls_tools.TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Agent finished — extract JSON from text block
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    text = block.text.strip()
                    start, end = text.find('{'), text.rfind('}') + 1
                    if start >= 0 and end > start:
                        try:
                            return json.loads(text[start:end]), trace
                        except json.JSONDecodeError:
                            pass
            return {"error": "Agent did not return valid JSON", "raw": str(response.content)}, trace

        # Agent wants to call tools
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = mls_tools.dispatch(block.name, block.input)

                    trace.append({
                        "turn": turn + 1,
                        "tool": block.name,
                        "input": block.input,
                        "result_preview": result[:400] + "…" if len(result) > 400 else result,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    # Safety: hit turn limit — ask agent to wrap up
    messages.append({
        "role": "user",
        "content": "Max tool calls reached. Please return your JSON findings now based on what you have gathered."
    })
    response = client.messages.create(
        model=MODEL, max_tokens=2048,
        system=SYSTEM_PROMPT_AGENT,
        messages=messages,
    )
    for block in response.content:
        if hasattr(block, 'text'):
            text = block.text.strip()
            start, end = text.find('{'), text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end]), trace
                except json.JSONDecodeError:
                    pass

    return {"error": "Agent loop exhausted without producing findings"}, trace


# ── Brief writer ──────────────────────────────────────────────────────────────

def _write_brief(
    lead: dict,
    guard_result: guard.GuardResult,
    findings: dict,
    client: anthropic.Anthropic,
) -> str:
    """Second LLM call — writes the final Lead Brief from research findings."""
    prompt = f"""\
Write a Lead Brief for the following buyer lead.

## Lead metadata
- Lead ID:  {lead['lead_id']}
- Received: {lead['received_at']}
- Channel:  {lead['channel']}
- Buyer:    {lead['buyer_name']} | {lead['buyer_email']} | {lead.get('buyer_phone') or 'no phone'}

## Original message (verbatim)
{lead['message']}

## Guard layer flags
{json.dumps(guard_result.flags, indent=2) if guard_result.flags else 'None'}

## Agent research findings
{json.dumps(findings, indent=2)}

Write the Lead Brief now."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT_BRIEF_WRITER,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Public API ────────────────────────────────────────────────────────────────

def process_lead(lead: dict, client: anthropic.Anthropic) -> dict:
    """
    Full pipeline for one lead.
    Returns structured result dict containing guard output, tool trace,
    agent findings, and the final Lead Brief markdown.
    """
    lead_id = lead['lead_id']
    print(f"\n{'═' * 60}")
    print(f"  {lead_id} — {lead['buyer_name']}")
    print(f"{'═' * 60}")

    # Stage 1: Guard
    print("  [1/3] Guard layer …")
    guard_result = guard.run(lead)
    if guard_result.flags:
        for flag in guard_result.flags:
            print(f"         {flag[:75]}…")

    # Stage 2: Agentic research loop
    print("  [2/3] Agentic research loop …")
    findings, trace = _run_agent_loop(lead, guard_result, client)
    confidence = findings.get('confidence', '?')
    print(f"         {len(trace)} tool call(s) | confidence: {confidence}")
    for t in trace:
        print(f"         Turn {t['turn']}: {t['tool']}({list(t['input'].keys())})")

    # Stage 3: Brief generation
    print("  [3/3] Writing Lead Brief …")
    brief_md = _write_brief(lead, guard_result, findings, client)
    print(f"         Done ✓  ({len(brief_md)} chars)")

    return {
        "lead_id": lead_id,
        "buyer_name": lead['buyer_name'],
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "guard": {
            "flags": guard_result.flags,
            "injection_detected": guard_result.injection_detected,
            "impossible_budget": guard_result.impossible_budget,
            "anonymous_lead": guard_result.anonymous_lead,
        },
        "agent_findings": findings,
        "tool_trace": trace,
        "lead_brief": brief_md,
    }
