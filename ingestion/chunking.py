"""
Three chunking strategies, deliberately kept separate so you can A/B them in
the eval harness and report a real comparison rather than picking one blind.

1. fixed_size_chunks          — naive baseline: split by token count with
                                  overlap, ignoring transcript structure.
                                  This is what most RAG tutorials do by
                                  default.
2. speaker_turn_chunks         — respects natural speaker-turn boundaries,
                                  merges short turns up to a target size, NO
                                  overlap between chunks. The original
                                  hypothesis: turn-aware chunking should
                                  score higher on faithfulness since it
                                  never chops an answer mid-sentence.
3. speaker_turn_chunks_overlap — same turn-respecting logic as #2, but adds
                                  a token-overlap window between consecutive
                                  chunks, matching fixed_size's overlap
                                  design. Added after the first benchmark
                                  run showed fixed_size unexpectedly beating
                                  plain speaker_turn on Recall@k — this
                                  isolates whether that gap was really about
                                  overlap (redundant coverage of boundary
                                  text) or something else about the chunking
                                  approach itself.

Run all three, embed each, retrieve with each, and let eval/llm_judge.py
tell you which actually wins on faithfulness — don't assume.
"""

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path

import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    company: str
    quarter: str
    section: str
    speaker_role: str
    text: str
    token_count: int
    strategy: str


def token_len(text: str) -> int:
    return len(ENCODER.encode(text))


def fixed_size_chunks(segments: list[dict], target_tokens=300, overlap_tokens=50) -> list[Chunk]:
    """Naive baseline: concatenate all segment text, then slide a fixed
    token-count window across it, ignoring speaker boundaries entirely."""
    full_text = " ".join(s["text"] for s in segments)
    tokens = ENCODER.encode(full_text)
    chunks = []
    start = 0
    idx = 0
    company = segments[0]["company"] if segments else "unknown"
    quarter = segments[0]["quarter"] if segments else "unknown"

    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_text = ENCODER.decode(tokens[start:end])
        chunks.append(
            Chunk(
                chunk_id=f"{company}_{quarter}_fixed_{idx}",
                company=company,
                quarter=quarter,
                section="mixed",
                speaker_role="mixed",
                text=chunk_text,
                token_count=end - start,
                strategy="fixed_size",
            )
        )
        idx += 1
        start += target_tokens - overlap_tokens

    return chunks


def split_long_turn_by_sentence(
    speaker_prefix: str, text: str, target_tokens: int
) -> list[str]:
    """
    Splits a single speaker turn's text at SENTENCE boundaries into pieces
    each under target_tokens, never cutting mid-sentence. Used when one
    turn (e.g. a long executive monologue) is already bigger than the
    target on its own — without this, that whole turn becomes one oversized
    chunk (we saw a real 3,341-token chunk from this exact case), which
    dilutes retrieval: a question about one detail buried in a long answer
    would retrieve the entire answer as a single, unfocused vector.
    Each resulting sub-chunk keeps the speaker prefix, so it's still
    self-contained and attributable on its own.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current, current_tokens = [], [], 0
    prefix_tokens = token_len(speaker_prefix)

    for sentence in sentences:
        sentence_tokens = token_len(sentence)
        if current and current_tokens + sentence_tokens + prefix_tokens > target_tokens:
            pieces.append(speaker_prefix + " " + " ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append(speaker_prefix + " " + " ".join(current))
    return pieces


def speaker_turn_chunks(segments: list[dict], target_tokens=300) -> list[Chunk]:
    """Merge consecutive same-section speaker turns up to ~target_tokens,
    but never split a single speaker turn across two chunks — UNLESS that
    turn is already longer than target_tokens on its own (e.g. a long
    executive monologue), in which case it gets split at sentence
    boundaries via split_long_turn_by_sentence, never mid-sentence."""
    chunks = []
    buffer_text, buffer_tokens = [], 0
    idx = 0
    current_section, current_role = None, None

    def flush(company, quarter):
        nonlocal idx, buffer_text, buffer_tokens
        if buffer_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{company}_{quarter}_turn_{idx}",
                    company=company,
                    quarter=quarter,
                    section=current_section or "mixed",
                    speaker_role=current_role or "mixed",
                    text=" ".join(buffer_text),
                    token_count=buffer_tokens,
                    strategy="speaker_turn",
                )
            )
            idx += 1
        buffer_text, buffer_tokens = [], 0

    for seg in segments:
        seg_tokens = token_len(seg["text"])
        speaker_prefix = f"[{seg['speaker_name']} ({seg['speaker_role']})]:"

        if seg_tokens > target_tokens:
            # This single turn is already bigger than the target — flush
            # whatever's buffered first, then emit this turn as its own
            # sentence-split sub-chunks rather than merging it with anything.
            flush(seg["company"], seg["quarter"])
            current_section, current_role = seg["section"], seg["speaker_role"]
            for piece_text in split_long_turn_by_sentence(speaker_prefix, seg["text"], target_tokens):
                chunks.append(
                    Chunk(
                        chunk_id=f"{seg['company']}_{seg['quarter']}_turn_{idx}",
                        company=seg["company"],
                        quarter=seg["quarter"],
                        section=seg["section"],
                        speaker_role=seg["speaker_role"],
                        text=piece_text,
                        token_count=token_len(piece_text),
                        strategy="speaker_turn",
                    )
                )
                idx += 1
            continue

        would_exceed = buffer_tokens + seg_tokens > target_tokens
        section_changed = current_section is not None and seg["section"] != current_section

        if would_exceed or section_changed:
            flush(seg["company"], seg["quarter"])

        current_section = seg["section"]
        current_role = seg["speaker_role"]
        buffer_text.append(f"{speaker_prefix} {seg['text']}")
        buffer_tokens += seg_tokens

    if segments:
        flush(segments[-1]["company"], segments[-1]["quarter"])

    return chunks


def build_atomic_pieces(segments: list[dict], target_tokens: int) -> list[dict]:
    """
    Breaks every segment into one or more speaker-tagged pieces, each
    already under target_tokens (short turns become one piece as-is; long
    turns are pre-split via split_long_turn_by_sentence). This separates
    "how do we atomize the transcript" from "how do we assemble atoms into
    chunks with/without overlap", which is what lets
    speaker_turn_chunks_overlap reuse the same sentence-safe splitting logic
    without duplicating it.
    """
    pieces = []
    for seg in segments:
        speaker_prefix = f"[{seg['speaker_name']} ({seg['speaker_role']})]:"
        seg_tokens = token_len(seg["text"])
        if seg_tokens > target_tokens:
            for piece_text in split_long_turn_by_sentence(speaker_prefix, seg["text"], target_tokens):
                pieces.append({
                    "text": piece_text, "tokens": token_len(piece_text),
                    "section": seg["section"], "speaker_role": seg["speaker_role"],
                    "company": seg["company"], "quarter": seg["quarter"],
                })
        else:
            full_text = f"{speaker_prefix} {seg['text']}"
            pieces.append({
                "text": full_text, "tokens": token_len(full_text),
                "section": seg["section"], "speaker_role": seg["speaker_role"],
                "company": seg["company"], "quarter": seg["quarter"],
            })
    return pieces


def speaker_turn_chunks_overlap(segments: list[dict], target_tokens=300, overlap_tokens=50) -> list[Chunk]:
    """
    Same speaker-turn-respecting logic as speaker_turn_chunks, but each
    chunk is prefixed with the last `overlap_tokens` worth of text from the
    PREVIOUS chunk — mirroring fixed_size_chunks' overlap window. Built as a
    separate function (not a parameter on speaker_turn_chunks) specifically
    so the original, already-benchmarked "_turns.json" output — and every
    gold_chunk_id in eval/gold_qa.json that points into it — stays exactly
    as-is. This exists to test a specific hypothesis from the first
    benchmark run: fixed_size chunking scored higher on Recall@k than
    speaker_turn chunking, and the working theory was that this came from
    fixed_size's overlap (duplicating boundary-adjacent text into two
    chunks, doubling its odds of being retrieved) rather than from
    fixed_size producing genuinely better chunks. If this overlap-added
    version closes or reverses the recall gap, that theory is confirmed; if
    fixed_size still wins, the effect is coming from something else.
    """
    if not segments:
        return []
    pieces = build_atomic_pieces(segments, target_tokens)
    chunks = []
    idx = 0
    buffer_text, buffer_tokens = [], 0
    current_section, current_role = None, None
    carry_over = ""

    def flush(company, quarter):
        nonlocal idx, buffer_text, buffer_tokens, carry_over
        if not buffer_text:
            return
        full_text = " ".join(buffer_text)
        chunks.append(
            Chunk(
                chunk_id=f"{company}_{quarter}_turnoverlap_{idx}",
                company=company, quarter=quarter,
                section=current_section or "mixed",
                speaker_role=current_role or "mixed",
                text=full_text, token_count=buffer_tokens,
                strategy="speaker_turn_overlap",
            )
        )
        idx += 1
        tail_tokens = ENCODER.encode(full_text)[-overlap_tokens:]
        carry_over = ENCODER.decode(tail_tokens) if tail_tokens else ""
        buffer_text, buffer_tokens = [], 0

    for piece in pieces:
        would_exceed = buffer_tokens + piece["tokens"] > target_tokens
        section_changed = current_section is not None and piece["section"] != current_section

        if would_exceed or section_changed:
            flush(piece["company"], piece["quarter"])
            if carry_over:
                buffer_text.append(carry_over)
                buffer_tokens += token_len(carry_over)

        current_section = piece["section"]
        current_role = piece["speaker_role"]
        buffer_text.append(piece["text"])
        buffer_tokens += piece["tokens"]

    flush(pieces[-1]["company"], pieces[-1]["quarter"])
    return chunks


def run_both_strategies(parsed_json_path: Path, output_dir: Path):
    segments = json.loads(parsed_json_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = fixed_size_chunks(segments)
    turns = speaker_turn_chunks(segments)
    turns_overlap = speaker_turn_chunks_overlap(segments)

    stem = parsed_json_path.stem
    (output_dir / f"{stem}_fixed.json").write_text(
        json.dumps([asdict(c) for c in fixed], indent=2)
    )
    (output_dir / f"{stem}_turns.json").write_text(
        json.dumps([asdict(c) for c in turns], indent=2)
    )
    (output_dir / f"{stem}_turns_overlap.json").write_text(
        json.dumps([asdict(c) for c in turns_overlap], indent=2)
    )
    print(f"{stem}: fixed_size -> {len(fixed)} chunks, speaker_turn -> {len(turns)} chunks, "
          f"speaker_turn_overlap -> {len(turns_overlap)} chunks")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parsed transcript JSON from parse_transcript.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_both_strategies(Path(args.input), Path(args.output))
