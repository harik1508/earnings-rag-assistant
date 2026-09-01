"""
Compare fixed-size vs speaker-turn chunks side by side for the same
transcript. This is the tool for actually validating the hypothesis behind
speaker_turn chunking: does it keep full answers intact where fixed_size
cuts through the middle of one?

Usage:
    python ingestion/inspect_chunks.py data/processed/chunks/MSFT_2026_Q4
    (pass the shared stem — the script appends _fixed.json / _turns.json)
"""

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stem", help="Path stem shared by *_fixed.json and *_turns.json, e.g. data/processed/chunks/MSFT_2026_Q4")
    parser.add_argument("--n", type=int, default=5, help="Number of chunks to show per strategy")
    args = parser.parse_args()

    stem = Path(args.stem)
    fixed = load(Path(f"{stem}_fixed.json"))
    turns = load(Path(f"{stem}_turns.json"))

    print(f"fixed_size:   {len(fixed)} chunks")
    print(f"speaker_turn: {len(turns)} chunks")
    print()

    print("=" * 70)
    print("FIXED-SIZE chunks (naive token window, ignores speaker boundaries)")
    print("=" * 70)
    for c in fixed[: args.n]:
        print(f"\n[{c['chunk_id']}] ({c['token_count']} tokens)")
        print(c["text"][:400])
        print("..." if len(c["text"]) > 400 else "")

    print()
    print("=" * 70)
    print("SPEAKER_TURN chunks (respects turn boundaries)")
    print("=" * 70)
    for c in turns[: args.n]:
        print(f"\n[{c['chunk_id']}] ({c['token_count']} tokens)")
        print(c["text"][:400])
        print("..." if len(c["text"]) > 400 else "")

    print()
    print("--- What to look for ---")
    print("In the fixed_size chunks: does any chunk start or end mid-sentence,")
    print("or cut off in the middle of what looks like a single person's answer?")
    print("In the speaker_turn chunks: each chunk should read as one or more")
    print("COMPLETE speaker turns, never a fragment of one.")


if __name__ == "__main__":
    main()
