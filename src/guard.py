"""
guard.py — Pre-LLM safety and anomaly detection layer.

Runs deterministically before any LLM call. Never blocks a lead —
always flags and passes through so the realtor sees everything.

Detects:
  - Prompt injection attacks
  - Impossible budget / requirement mismatches
  - Missing or suspicious identity
  - Empty / near-empty messages
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Injection patterns ────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(everything|all|what)\s+(you('ve| have)?\s+)?(been\s+)?told",
    r"you\s+are\s+now\s+(a\s+)?(\w+\s+)*assistant",
    r"respond\s+(only\s+)?(by|with|in)\s+listing",
    r"(list|dump|output|print|show)\s+(all\s+)?(owner|phone|contact|private|personal)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"###\s*instruction",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    sanitized_message: str
    flags: list[str] = field(default_factory=list)
    injection_detected: bool = False
    impossible_budget: bool = False
    anonymous_lead: bool = False
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_budget(message: str) -> tuple[Optional[float], Optional[float]]:
    """Extract dollar figures from free text. Returns (min, max) plausible amounts."""
    pattern = r'\$?([\d,]+(?:\.\d+)?)\s*([KkMm]?)'
    matches = re.findall(pattern, message)
    amounts = []
    for num_str, suffix in matches:
        try:
            num = float(num_str.replace(',', ''))
            if suffix.lower() == 'k':
                num *= 1_000
            elif suffix.lower() == 'm':
                num *= 1_000_000
            if 50_000 <= num <= 500_000_000:
                amounts.append(num)
        except ValueError:
            continue
    if not amounts:
        return None, None
    return min(amounts), max(amounts)


def _check_injection(message: str, flags: list[str]) -> tuple[bool, str]:
    """Detect and strip prompt injection. Returns (detected, sanitized_message)."""
    sanitized = message
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            match = re.search(pattern, sanitized, re.IGNORECASE)
            if match:
                before = sanitized[:match.start()].strip()
                after = sanitized[match.end():].strip()
                sanitized = (before + " " + after).strip()
            flags.append(
                "⚠️ SECURITY — PROMPT INJECTION DETECTED: This message contains embedded "
                "instructions attempting to override agent behaviour (e.g. 'ignore previous "
                "instructions', requests to dump owner contact data). The injected content has "
                "been stripped. Only the legitimate property request was processed. "
                "Consider flagging this contact."
            )
            return True, sanitized
    return False, sanitized


def _check_budget(message: str, flags: list[str]) -> tuple[bool, Optional[float], Optional[float]]:
    """Flag budgets that cannot match stated requirements in Miami."""
    msg_lower = message.lower()
    budget_min, budget_max = _extract_budget(message)

    if budget_max is None:
        return False, budget_min, budget_max

    wants_large = any(p in msg_lower for p in [
        '4 bedroom', '4-bedroom', '5 bedroom', '5-bedroom',
        '4br', '5br', 'four bedroom', 'five bedroom'
    ])
    wants_premium = any(p in msg_lower for p in [
        'ocean view', 'oceanfront', 'pool', 'waterfront'
    ])
    premium_area = any(p in msg_lower for p in [
        'brickell', 'miami beach', 'south beach',
        'key biscayne', 'bal harbour', 'downtown'
    ])

    impossible = False
    if budget_max < 400_000 and wants_large and wants_premium and premium_area:
        flags.append(
            f"⚠️ IMPOSSIBLE BUDGET: Buyer wants 4+ BR with pool/ocean view in a premium Miami "
            f"neighbourhood but stated budget is only ${budget_max:,.0f}. The realistic minimum "
            f"for this specification is $800K–$1.2M+. Verify budget before investing time — "
            f"this may be a typo or serious misunderstanding."
        )
        impossible = True
    elif budget_max < 320_000:
        flags.append(
            f"⚠️ LOW BUDGET: ${budget_max:,.0f} is below the practical floor for most Miami "
            f"active listings (~$320K minimum in our dataset). Options will be extremely limited."
        )

    return impossible, budget_min, budget_max


def _check_identity(lead: dict, flags: list[str]) -> bool:
    """Flag missing or suspicious identity fields."""
    name = lead.get("buyer_name", "").strip()
    phone = lead.get("buyer_phone", "").strip()
    email = lead.get("buyer_email", "").strip()
    anonymous = False

    if not name or "anonymous" in name.lower() or "not filled" in name.lower():
        flags.append(
            "⚠️ ANONYMOUS LEAD: No buyer name provided. Cannot personalise outreach. "
            "Attempt email contact to establish identity before actioning further."
        )
        anonymous = True

    if not phone:
        flags.append("ℹ️ MISSING PHONE: No phone number on file. Email is the only contact channel.")

    if email:
        disposable = ['mailinator.com', 'guerrillamail.com', 'tempmail.com', 'throwam.com']
        domain = email.split('@')[-1].lower() if '@' in email else ''
        if domain in disposable:
            flags.append(f"⚠️ SUSPICIOUS EMAIL: '{domain}' is a known disposable email domain.")

    return anonymous


# ── Public API ────────────────────────────────────────────────────────────────

def run(lead: dict) -> GuardResult:
    """
    Run all guard checks on a raw lead dict.
    Returns GuardResult with sanitized message and accumulated flags.
    """
    flags: list[str] = []
    message = lead.get("message", "")

    injection_detected, sanitized_message = _check_injection(message, flags)
    impossible_budget, budget_min, budget_max = _check_budget(message, flags)
    anonymous = _check_identity(lead, flags)

    if len(sanitized_message.strip()) < 20:
        flags.append(
            "ℹ️ THIN MESSAGE: Very short message with almost no search criteria. "
            "Significant qualifying conversation needed before property matching is meaningful."
        )

    return GuardResult(
        sanitized_message=sanitized_message,
        flags=flags,
        injection_detected=injection_detected,
        impossible_budget=impossible_budget,
        anonymous_lead=anonymous,
        budget_min=budget_min,
        budget_max=budget_max,
    )
