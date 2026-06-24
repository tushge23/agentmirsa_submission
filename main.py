#!/usr/bin/env python3
"""
main.py — CLI entry point for the Buyer Lead Intake Agent.

Usage:
    python main.py                           # process all 12 leads
    python main.py --lead LEAD-2026-001      # single lead
    python main.py --lead LEAD-2026-001 LEAD-2026-006

Output:
    output/briefs/LEAD-XXXX-XXX.md    individual Lead Brief (Markdown)
    output/briefs/LEAD-XXXX-XXX.json  full result with tool trace
    output/all_briefs.md              all 12 briefs in one file
    logs/run_TIMESTAMP.json           complete run log
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from src import agent


def load_leads(data_dir: Path) -> list[dict]:
    path = data_dir / "sample_buyer_inquiries.json"
    with open(path) as f:
        return json.load(f)


def save_result(result: dict, output_dir: Path) -> None:
    briefs_dir = output_dir / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    lead_id = result["lead_id"]

    md_path = briefs_dir / f"{lead_id}.md"
    with open(md_path, "w") as f:
        f.write(f"# Lead Brief — {lead_id}\n")
        f.write(f"*Generated: {result['processed_at']}*\n\n")
        f.write(result["lead_brief"])

    json_path = briefs_dir / f"{lead_id}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentMira — Buyer Lead Intake Agent")
    parser.add_argument("--lead", nargs="+", metavar="LEAD_ID",
                        help="Process specific lead ID(s). Default: all.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    root = Path(__file__).parent
    data_dir = root / args.data_dir
    output_dir = root / args.output_dir

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("       export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    all_leads = load_leads(data_dir)

    leads = (
        [l for l in all_leads if l["lead_id"] in args.lead]
        if args.lead else all_leads
    )

    print(f"\n{'═' * 60}")
    print(f"  AgentMira — Buyer Lead Intake Agent")
    print(f"  Processing {len(leads)} lead(s) …")
    print(f"{'═' * 60}")

    results, failed, brief_parts = [], [], []

    for lead in leads:
        try:
            result = agent.process_lead(lead, client)
            save_result(result, output_dir)
            results.append(result)
            brief_parts.append(
                f"\n---\n\n# {result['lead_id']} — {result['buyer_name']}\n\n"
                f"{result['lead_brief']}\n"
            )
        except Exception as e:
            print(f"  ERROR on {lead['lead_id']}: {e}")
            failed.append({"lead_id": lead["lead_id"], "error": str(e)})

    # Combined brief file
    if brief_parts:
        combined = output_dir / "all_briefs.md"
        with open(combined, "w") as f:
            f.write("# AgentMira — All Lead Briefs\n")
            f.write(f"*Generated: {datetime.now(timezone.utc).isoformat()}*\n")
            f.write("".join(brief_parts))
        print(f"\n  All briefs → {combined}")

    # Run log
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path = logs_dir / f"run_{ts}.json"
    with open(log_path, "w") as f:
        json.dump({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "model": "claude-sonnet-4-6",
            "leads_processed": len(results),
            "leads_failed": len(failed),
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"  Run log    → {log_path}")

    print(f"\n{'═' * 60}")
    print(f"  DONE  {len(results)} succeeded  |  {len(failed)} failed")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
