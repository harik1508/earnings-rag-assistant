"""
Quick spot-check tool: prints the first N parsed segments from a transcript
JSON so you can eyeball whether speaker names/roles/text actually look right,
not just whether the role-count totals look plausible.

Usage:
    python ingestion/inspect_segments.py data/processed/JPM_2026_Q1.json
    python ingestion/inspect_segments.py data/processed/JPM_2026_Q1.json --n 20
    python ingestion/inspect_segments.py data/processed/JPM_2026_Q1.json --role analyst
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a parsed transcript JSON file")
    parser.add_argument("--n", type=int, default=10, help="Number of segments to show")
    parser.add_argument("--role", default=None, help="Only show segments with this speaker_role")
    args = parser.parse_args()

    segments = json.loads(Path(args.path).read_text())
    if args.role:
        segments = [s for s in segments if s["speaker_role"] == args.role]

    for s in segments[: args.n]:
        role = s["speaker_role"].ljust(10)
        name = s["speaker_name"].ljust(20)
        text = s["text"][:70]
        print(f"[{role}] {name} | {text}")

    print(f"\nShowing {min(args.n, len(segments))} of {len(segments)} segments"
          + (f" (filtered to role={args.role})" if args.role else ""))


if __name__ == "__main__":
    main()
