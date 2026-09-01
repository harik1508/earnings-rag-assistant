"""
Parses raw earnings call transcripts into a structured, speaker-tagged format.

Why speaker tagging matters for this project:
Earnings calls have two very different voices — prepared remarks from
executives (confident, rehearsed, often vague on specifics) and live Q&A
(executives responding to analyst questions, sometimes evasively). Our eval
harness later needs to distinguish these, so we tag every text segment with
its speaker role at ingestion time rather than trying to infer it later.

Usage:
    python parse_transcript.py --input data/raw/MSFT_2025_Q4.txt --output data/processed/
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pdfplumber


# Common exec title keywords used to classify a speaker line as "executive"
# vs "analyst". This is a heuristic, not perfect — flag mis-tags you spot
# while building your gold eval set and refine this list.
EXEC_TITLE_KEYWORDS = [
    "chief executive", "ceo", "chief financial", "cfo", "president",
    "chairman", "chief operating", "coo", "investor relations",
]

# The same person is sometimes named differently across quarters by the same
# publisher (e.g. "Amy Hood" in one transcript, "Amy E. Hood" in another).
# Left as-is, that breaks any downstream grouping/filtering by speaker across
# quarters — exactly the kind of thing a cross-quarter gold Q&A question
# would silently trip over. Normalize known aliases to one canonical name at
# ingestion time rather than downstream, where every consumer would need to
# remember to do it. Extend this map as you spot new aliases while building
# your gold set — this list won't be exhaustive on the first pass.
NAME_ALIASES = {
    "Amy E. Hood": "Amy Hood",
    "Brian Defoe": "Brian DeFoe",  # capitalization drift, same person
}


def normalize_speaker_name(name: str) -> str:
    return NAME_ALIASES.get(name, name)


# Conference-call operators are sometimes labeled "Operator" and sometimes
# labeled with their actual first name (e.g. "Sarah:") depending on which
# conferencing service ran the call. A first-name-only label can't be
# distinguished from a stray single word by name alone, so we detect the
# operator by what they characteristically SAY instead. Kept deliberately
# narrow to phrases that are essentially operator-exclusive — an earlier,
# looser version included "welcome to" and "please poll for questions",
# both of which the investor-relations host also says in their own
# introduction, which mislabeled the host as the operator. Broad phrases
# feel like they'd catch more cases, but each one added is a new way to
# mislabel a real speaker; narrow and check the output beats broad and hope.
OPERATOR_PHRASE_PATTERN = re.compile(
    r"next question comes from|your line is open|star.{0,15}number one|"
    r"this concludes|you may disconnect|"
    r"final question comes from|last question (?:will come|comes) from",
    re.IGNORECASE,
)

# Matches lines like "John Smith - Chief Financial Officer" (used by some
# sources, e.g. Seeking Alpha). Deliberately strict: title text must not
# contain digits or "$", and accepts hyphen, en-dash, or em-dash as the
# separator (different transcript exports use different dash characters for
# what is visually "the same" line) — this stops it from misfiring on
# stat-bullet lines like "Revenue -- $90.0 billion..." which use a double
# dash and contain numbers. An earlier looser version of this pattern
# matched those bullets as if they were speaker lines, which silently
# corrupted every downstream segment — a reminder to always inspect actual
# parsed output rather than trusting that "it ran without errors" means it
# ran correctly.
SPEAKER_LINE_PATTERN = re.compile(
    r"^([A-Z][a-zA-Z.\-' ]{2,40})\s[-–—]\s([A-Za-z][a-zA-Z,.&' ]{2,60})$"
)

# Matches Motley Fool style dialogue lines: "**Satya Nadella:**" or "Satya Nadella:"
# with no inline title — the title has to be looked up separately from the
# CALL PARTICIPANTS roster printed once near the top of the transcript.
DIALOGUE_NAME_ONLY_PATTERN = re.compile(
    r"^\*{0,2}([A-Z][a-zA-Z.\-' ]{2,40})\*{0,2}:\s*(.*)$"
)

# Some transcripts spell the role out explicitly instead of relying on the
# roster, e.g. "Analyst (Michael Lasser): Good evening...". The name here
# sits inside parentheses, which DIALOGUE_NAME_ONLY_PATTERN's character
# class doesn't allow, so without this dedicated pattern the whole line
# fails to match anything and silently merges into whatever segment came
# before it — exactly what happened before this pattern was added.
ANALYST_LABELED_PATTERN = re.compile(
    r"^Analyst\s*\(([^)]+)\)\s*:\s*(.*)$"
)

# Matches roster lines like "- Chief Financial Officer - Amy Hood" or
# "Chairman and Chief Executive Officer — Satya Nadella" (em-dash variant,
# which is what several MSFT quarters actually use, unlike the plain hyphen
# seen in the Q4 file — same lesson as SPEAKER_LINE_PATTERN above).
ROSTER_LINE_PATTERN = re.compile(
    r"^-?\s*(.+?)\s*[-–—]\s*([A-Z][a-zA-Z.\-' ]{2,40})$"
)


def build_roster(raw_text: str) -> tuple[dict[str, str], str | None]:
    """
    Parses the 'Call participants' section (Motley Fool format, case and
    dash-style varies by quarter) into a {speaker_name: role} lookup, plus
    the CFO's name specifically if one is found. The CFO name is returned
    separately because some transcripts open with the CFO's prepared remarks
    with no speaker label at all (assuming the reader already knows who's
    talking) — see the CFO fallback in segment_by_speaker. Falls back to an
    empty dict / None if the section isn't found — the caller should then
    rely on SPEAKER_LINE_PATTERN instead.
    """
    roster = {}
    cfo_name = None
    # End marker is whichever of these section headers appears first — the
    # set of headers Motley Fool includes varies by quarter (e.g. some
    # quarters skip straight from participants to "Risks" with no
    # "Takeaways" section at all), so we list every header we've seen
    # rather than assuming one specific one is always present.
    match = re.search(
        r"Call participants(.*?)(?:\n##|\nTakeaways|\nRisks|\nSummary|"
        r"\nIndustry Glossary|\nFull Conference Call Transcript)",
        raw_text, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return roster, cfo_name

    for line in match.group(1).splitlines():
        line = line.strip()
        roster_match = ROSTER_LINE_PATTERN.match(line)
        if roster_match:
            title_text, name = roster_match.group(1), roster_match.group(2)
            name = normalize_speaker_name(name.strip())
            roster[name] = classify_speaker(title_text)
            if "chief financial" in title_text.lower() or "cfo" in title_text.lower():
                cfo_name = name
    return roster, cfo_name


@dataclass
class TranscriptSegment:
    company: str
    quarter: str
    speaker_name: str
    speaker_role: str  # "executive" | "analyst" | "operator" | "unknown"
    section: str        # "prepared_remarks" | "qa"
    text: str
    segment_id: str


def parse_filename(path: Path) -> tuple[str, str]:
    """Expects TICKER_YEAR_QN.txt, e.g. MSFT_2025_Q4.txt"""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Filename '{path.name}' doesn't match TICKER_YEAR_QN convention"
        )
    company, year, quarter = parts
    return company, f"{year}_{quarter}"


def load_raw_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        text_chunks = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_chunks.append(page.extract_text() or "")
        return "\n".join(text_chunks)
    return path.read_text(encoding="utf-8", errors="ignore")


def classify_speaker(title_text: str) -> str:
    lowered = title_text.lower()
    if "operator" in lowered:
        return "operator"
    if any(keyword in lowered for keyword in EXEC_TITLE_KEYWORDS):
        return "executive"
    # General fallback: any title containing both "chief" and "officer"
    # is a C-suite executive, regardless of which specific function (Chief
    # Commercial Officer, Chief Marketing Officer, Chief Revenue Officer,
    # etc.). The keyword list above only enumerated the handful of titles
    # seen in earlier transcripts (CEO/CFO/COO) — that's an incomplete list
    # by construction, since companies invent new "Chief X Officer" titles
    # constantly. A pattern-based rule generalizes where an enumerated list
    # can't; this is what caught Southwest's "Chief Commercial Officer"
    # falling through as unclassified.
    if "chief" in lowered and "officer" in lowered:
        return "executive"
    if "analyst" in lowered:
        return "analyst"
    return "unknown"


def strip_front_matter(raw_text: str) -> str:
    """
    Motley Fool (and similar) transcripts open with editorial front matter —
    DATE, CALL PARTICIPANTS, TAKEAWAYS, RISKS, SUMMARY, INDUSTRY GLOSSARY —
    before the actual "Full Conference Call Transcript" begins. That front
    matter uses bullet lines like "Revenue -- $90.0 billion..." which can be
    mistaken for dialogue by a naive parser. We cut everything before the
    real transcript starts so only actual speaker turns get segmented.
    Falls back to the full text unchanged if no such header is found (e.g.
    a source that has no editorial front matter at all).
    """
    match = re.search(
        r"Full Conference Call Transcript|^Prepared Remarks:?$",
        raw_text, re.IGNORECASE | re.MULTILINE,
    )
    return raw_text[match.end():] if match else raw_text


def split_prepared_vs_qa(raw_text: str) -> tuple[str, str]:
    """
    Splits the transcript into prepared remarks vs Q&A using common section
    markers. Falls back to a conversational handoff phrase (e.g. "let's go
    to Q&A"), then to treating the whole doc as one block if neither is
    found — in that last case, section labels will all read "prepared_
    remarks", which is fine since eval mainly cares about speaker_role, not
    this label; flag these files if you want the section split refined.
    """
    qa_markers = [
        r"question-and-answer session",
        r"questions?\s*&\s*answers?",
        r"q\s*&\s*a session",
        r"let'?s go to q\s*&\s*a",
        r"move (?:over )?to q\s*&\s*a",
    ]
    for marker in qa_markers:
        match = re.search(marker, raw_text, re.IGNORECASE)
        if match:
            return raw_text[: match.start()], raw_text[match.start():]
    return raw_text, ""


def segment_by_speaker(
    section_text: str, company: str, quarter: str, section_label: str,
    roster: dict[str, str] | None = None, default_speaker: str | None = None,
) -> list[TranscriptSegment]:
    """
    Tries SPEAKER_LINE_PATTERN first (inline title, e.g. Seeking Alpha style).
    Falls back to DIALOGUE_NAME_ONLY_PATTERN + roster lookup (Motley Fool
    style) when no inline title is present.

    default_speaker: if the section's opening lines have no speaker label at
    all (some transcripts start the CFO's remarks without naming them, since
    the reader is assumed to already know), attribute those leading lines to
    this name instead of "Unknown". Only applied before the first real
    speaker match is found — once any speaker line matches, we trust the
    transcript's own labeling from then on.
    """
    roster = roster or {}
    segments = []
    if default_speaker and default_speaker in roster:
        current_speaker, current_role = default_speaker, roster[default_speaker]
    else:
        current_speaker, current_role = "Unknown", "unknown"
    buffer = []

    def flush(idx: int):
        if buffer:
            segments.append(
                TranscriptSegment(
                    company=company,
                    quarter=quarter,
                    speaker_name=current_speaker,
                    speaker_role=current_role,
                    section=section_label,
                    text=" ".join(buffer).strip(),
                    segment_id=f"{company}_{quarter}_{section_label}_{idx}",
                )
            )

    idx = 0
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue

        analyst_labeled_match = ANALYST_LABELED_PATTERN.match(line)
        inline_match = SPEAKER_LINE_PATTERN.match(line)
        name_only_match = DIALOGUE_NAME_ONLY_PATTERN.match(line)

        if analyst_labeled_match:
            flush(idx)
            idx += 1
            buffer = []
            current_speaker = normalize_speaker_name(analyst_labeled_match.group(1).strip())
            current_role = "analyst"
            remainder = analyst_labeled_match.group(2).strip()
            if remainder:
                buffer.append(remainder)
        elif inline_match:
            flush(idx)
            idx += 1
            buffer = []
            current_speaker = normalize_speaker_name(inline_match.group(1).strip())
            title_text = inline_match.group(2).strip()
            current_role = classify_speaker(title_text)
            remainder = ""  # inline format has no trailing dialogue on the same line typically
        elif name_only_match:
            candidate = normalize_speaker_name(name_only_match.group(1).strip())
            remainder_text = name_only_match.group(2).strip()
            is_operator_label = candidate.lower() == "operator"
            # A single-word name whose line content is a classic operator
            # phrase (e.g. "Sarah: The next question comes from...") is also
            # treated as the operator — this is what catches conferencing
            # services that label the operator by first name instead of the
            # word "Operator". Checked before the two-word name requirement
            # below, since a first-name-only operator would otherwise fail
            # that check and get silently merged into the prior speaker.
            is_operator_by_phrase = bool(OPERATOR_PHRASE_PATTERN.search(remainder_text))
            # Require a first+last name shape (two capitalized words) before
            # treating an unlisted name as a new speaker — this is what
            # stops single capitalized words in ordinary prose (e.g. a
            # glossary term or a sentence starting "Note:") from being
            # misread as a speaker turn. Analysts are almost never listed in
            # the roster (Motley Fool only rosters the company's own
            # executives), so for them this two-word check is the main
            # safety net, not the roster.
            looks_like_person_name = bool(
                re.match(r"^[A-Z][a-zA-Z.\-']+(\s[A-Z][a-zA-Z.\-']+)+$", candidate)
            )
            is_operator = is_operator_label or is_operator_by_phrase
            if candidate in roster or is_operator or looks_like_person_name:
                flush(idx)
                idx += 1
                buffer = []
                current_speaker = "Operator" if is_operator else candidate
                current_role = (
                    "operator" if is_operator
                    else roster.get(candidate, "analyst")  # unlisted name in Q&A -> assume analyst
                )
                if remainder_text:
                    buffer.append(remainder_text)
            else:
                buffer.append(line)
        else:
            buffer.append(line)
    flush(idx)
    return [s for s in segments if len(s.text) > 20]  # drop noise lines


def parse_transcript_file(input_path: Path) -> list[TranscriptSegment]:
    company, quarter = parse_filename(input_path)
    raw_text = load_raw_text(input_path)
    roster, cfo_name = build_roster(raw_text)  # built from front matter, before we strip it
    transcript_only = strip_front_matter(raw_text)
    prepared, qa = split_prepared_vs_qa(transcript_only)

    # default_speaker only applies to prepared_remarks: that's the only
    # section that can plausibly open with unlabeled dialogue (the CFO's
    # opening lines before the first "Name:" tag appears). The Q&A section
    # always starts with an explicit operator/analyst label in every
    # transcript we've seen, so no default is passed there.
    segments = segment_by_speaker(
        prepared, company, quarter, "prepared_remarks", roster, default_speaker=cfo_name
    )
    segments += segment_by_speaker(qa, company, quarter, "qa", roster)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a single transcript file")
    parser.add_argument("--output", required=True, help="Output directory for parsed JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = parse_transcript_file(input_path)
    out_path = output_dir / f"{input_path.stem}.json"
    out_path.write_text(json.dumps([asdict(s) for s in segments], indent=2))

    print(f"Parsed {len(segments)} segments -> {out_path}")
    role_counts = {}
    for s in segments:
        role_counts[s.speaker_role] = role_counts.get(s.speaker_role, 0) + 1
    print(f"Speaker role breakdown: {role_counts}")


if __name__ == "__main__":
    main()
