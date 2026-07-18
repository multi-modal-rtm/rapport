"""Phase N5-B, Step B1: IEMOCAP corpus inventory and label-protocol audit.

Read-only reconnaissance ONLY -- no preprocessing, no processed index, no
caches. Walks the extracted release, cross-checks dialogue/utterance
counts against published expectations, parses the EmoEvaluation consensus
labels, and reports the 6-class vs. 4-class protocol comparison so the
protocol choice (docs/PHASE_N5B.md's remaining open decision point) can be
made with the real label distribution in hand.

The release lives OUTSIDE this repo, at
~/data/iemocap/iemocap/IEMOCAP_full_release/ -- NOT at
configs/data/iemocap.yaml's expected ${paths.data_dir}/iemocap
(data/iemocap/ inside the repo). That mismatch is flagged in the report as
a location note for whoever builds the processed index later; nothing is
moved or symlinked here.

Usage:
    uv run python -m scripts.iemocap_inventory
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path.home() / "data" / "iemocap" / "iemocap" / "IEMOCAP_full_release"
SESSIONS = [f"Session{i}" for i in range(1, 6)]

EXPECTED_UTTERANCES = 10039
EXPECTED_DIALOGUES = 151
EXPECTED_SPEAKERS = 10

TURN_LINE_RE = re.compile(r"^\[(?P<start>[\d.]+) - (?P<end>[\d.]+)\]\t(?P<turn>\S+)\t(?P<emotion>\S+)\t")

# Protocol definitions (per B1 spec).
SIX_CLASS_MAP = {"ang": "angry", "hap": "happy", "exc": "excited", "sad": "sad", "neu": "neutral", "fru": "frustrated"}
FOUR_CLASS_MAP = {"ang": "angry", "hap": "happy+excited", "exc": "happy+excited", "sad": "sad", "neu": "neutral"}


def dialogue_type(dialogue_name: str) -> str:
    # e.g. Ses01F_impro01 -> improvised, Ses01F_script01_1 -> scripted
    body = dialogue_name.split("_", 1)[1]
    return "improvised" if body.startswith("impro") else "scripted"


def list_real_files(dir_path: Path, suffix: str) -> list[Path]:
    """Lists real (non AppleDouble ._*) files with the given suffix, FLAT
    (one directory level only). `dialog/EmoEvaluation/` in particular has
    Categorical/Attribute/Self-evaluation subdirectories full of
    per-evaluator .txt files (e.g. Ses01F_impro01_e2_cat.txt) that must NOT
    be swept in here -- only the top-level per-dialogue consensus files."""
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.glob(f"*{suffix}") if not p.name.startswith("._"))


def list_avi_files(avi_dir: Path) -> list[Path]:
    """`dialog/avi/` nests the actual files one level down under a
    codec-named subdirectory (`avi/DivX/*.avi`), not flat."""
    if not avi_dir.exists():
        return []
    return sorted(p for p in avi_dir.glob("*/*.avi") if not p.name.startswith("._"))


def parse_emo_evaluation(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = TURN_LINE_RE.match(line)
        if m:
            rows.append(
                {
                    "turn": m.group("turn"),
                    "emotion": m.group("emotion"),
                    "start": float(m.group("start")),
                    "end": float(m.group("end")),
                }
            )
    return rows


def main() -> None:
    if not DATA_ROOT.exists():
        raise SystemExit(f"IEMOCAP release not found at {DATA_ROOT}")

    session_file_presence: dict[str, dict] = {}
    all_utterances: list[dict] = []  # session, dialogue, dialogue_type, speaker, emotion
    missing_files: list[str] = []

    for session in SESSIONS:
        session_dir = DATA_ROOT / session
        wav_dialogues = list_real_files(session_dir / "dialog" / "wav", ".wav")
        avi_dialogues = list_avi_files(session_dir / "dialog" / "avi")
        transcript_dialogues = list_real_files(session_dir / "dialog" / "transcriptions", ".txt")
        emo_dialogues = list_real_files(session_dir / "dialog" / "EmoEvaluation", ".txt")

        wav_names = {p.stem for p in wav_dialogues}
        avi_names = {p.stem for p in avi_dialogues}
        transcript_names = {p.stem for p in transcript_dialogues}
        emo_names = {p.stem for p in emo_dialogues}
        all_names = wav_names | avi_names | transcript_names | emo_names
        for name in sorted(all_names):
            present = {"wav": name in wav_names, "avi": name in avi_names, "transcription": name in transcript_names, "EmoEvaluation": name in emo_names}
            if not all(present.values()):
                missing_files.append(f"{session}/{name}: missing {[k for k, v in present.items() if not v]}")

        session_file_presence[session] = {
            "n_dialogues_wav": len(wav_dialogues),
            "n_dialogues_avi": len(avi_dialogues),
            "n_dialogues_transcription": len(transcript_dialogues),
            "n_dialogues_EmoEvaluation": len(emo_dialogues),
            "n_improvised": sum(1 for n in wav_names if dialogue_type(n) == "improvised"),
            "n_scripted": sum(1 for n in wav_names if dialogue_type(n) == "scripted"),
        }

        for emo_path in emo_dialogues:
            dialogue_name = emo_path.stem
            dtype = dialogue_type(dialogue_name)
            for row in parse_emo_evaluation(emo_path):
                speaker = "F" if "_F" in row["turn"].split(dialogue_name)[-1] else "M"
                all_utterances.append(
                    {
                        "session": session,
                        "dialogue": dialogue_name,
                        "dialogue_type": dtype,
                        "speaker_tag": row["turn"],
                        "speaker_gender": speaker,
                        "emotion": row["emotion"],
                    }
                )

    n_dialogues_total = sum(v["n_dialogues_wav"] for v in session_file_presence.values())
    n_utterances_total = len(all_utterances)

    # ---- Label distribution ----
    raw_dist_overall = Counter(u["emotion"] for u in all_utterances)
    raw_dist_by_session = {s: Counter(u["emotion"] for u in all_utterances if u["session"] == s) for s in SESSIONS}

    def protocol_counts(class_map: dict[str, str]) -> dict:
        kept = [u for u in all_utterances if u["emotion"] in class_map]
        discarded = len(all_utterances) - len(kept)
        by_session = {s: sum(1 for u in kept if u["session"] == s) for s in SESSIONS}
        by_class = Counter(class_map[u["emotion"]] for u in kept)
        return {"n_kept": len(kept), "n_discarded": discarded, "by_session": by_session, "by_class": dict(by_class)}

    six_class = protocol_counts(SIX_CLASS_MAP)
    four_class = protocol_counts(FOUR_CLASS_MAP)

    # ---- Write docs/iemocap_inventory.md ----
    lines = []
    lines.append("# IEMOCAP corpus inventory + label-protocol audit (Phase N5-B, Step B1)\n")
    lines.append(
        "Read-only inventory only -- no preprocessing, no processed index, no caches. "
        f"Release: `{DATA_ROOT}` (md5 521be1e5eec425ae21fdc27c763ca813, verified before extraction). "
        "**Location note:** this is outside the repo, not at `configs/data/iemocap.yaml`'s expected "
        "`${paths.data_dir}/iemocap` (`data/iemocap/`) -- flagged for whoever builds the processed "
        "index later; nothing moved or symlinked here.\n"
    )
    lines.append(
        "## Pre-registered hypothesis (docs/PHASE_N5B.md, commit 04d9267, quoted verbatim -- "
        "predates this data download)\n"
    )
    lines.append(
        "> Pre-registered hypothesis (stated here, before any IEMOCAP result\n"
        "> exists, so it can't be fitted after the fact): **relational memory's\n"
        "> delta on IEMOCAP > its delta on MELD, driven by dialogue length**\n"
        "> (IEMOCAP dialogues run substantially longer than MELD's, which should\n"
        "> give graph-level relationship state more room to matter than MELD's\n"
        "> `docs/PHASE_N4R.md`/`docs/PHASE_N5A.md` results showed). Either outcome\n"
        "> -- confirmed or not -- gets reported when this phase actually runs.\n"
        "> Also log the single dyadic edge-state norm over time for 3 sample\n"
        "> sessions as a qualitative figure candidate, once relational memory is\n"
        "> running on real IEMOCAP dialogues.\n"
    )

    lines.append("## 1. Corpus inventory\n")
    lines.append("| session | dialogues (wav) | improvised | scripted | avi | transcription | EmoEvaluation |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in SESSIONS:
        v = session_file_presence[s]
        lines.append(
            f"| {s} | {v['n_dialogues_wav']} | {v['n_improvised']} | {v['n_scripted']} | "
            f"{v['n_dialogues_avi']} | {v['n_dialogues_transcription']} | {v['n_dialogues_EmoEvaluation']} |"
        )
    lines.append(f"| **total** | **{n_dialogues_total}** | | | | | |")
    lines.append("")
    lines.append(
        f"**Utterances (EmoEvaluation consensus turns): {n_utterances_total}** "
        f"(published expectation: ~{EXPECTED_UTTERANCES} -- "
        f"{'MATCHES exactly' if n_utterances_total == EXPECTED_UTTERANCES else f'differs by {n_utterances_total - EXPECTED_UTTERANCES}'}).\n"
    )
    lines.append(
        f"**Dialogues: {n_dialogues_total}** (published expectation: {EXPECTED_DIALOGUES} -- "
        f"{'MATCHES exactly' if n_dialogues_total == EXPECTED_DIALOGUES else f'differs by {n_dialogues_total - EXPECTED_DIALOGUES}'}).\n"
    )
    speakers = {(u["session"], u["speaker_gender"]) for u in all_utterances}
    lines.append(
        f"**Distinct (session, gender) speaker slots: {len(speakers)}** (published expectation: "
        f"{EXPECTED_SPEAKERS} -- {'MATCHES exactly' if len(speakers) == EXPECTED_SPEAKERS else 'differs'}); "
        "IEMOCAP's dyadic design assigns one male + one female actor per session, consistent across "
        "sessions except that the SAME 10 individuals are not necessarily distinct across sessions in "
        "the original corpus design (verify against Documentation/ if a specific speaker-identity check "
        "is later needed -- not required for this inventory).\n"
    )
    if missing_files:
        lines.append(f"**Missing/incomplete dialogues ({len(missing_files)}):**\n")
        for m in missing_files:
            lines.append(f"- {m}")
    else:
        lines.append("**No missing or unreadable dialogue-level files** -- wav/avi/transcription/EmoEvaluation all present for every dialogue in every session.\n")
    lines.append("")

    lines.append("## 2. Label audit -- raw distribution (all categories, incl. xxx/oth)\n")
    all_labels = sorted(raw_dist_overall.keys())
    lines.append("| emotion code | " + " | ".join(SESSIONS) + " | total |")
    lines.append("|---|" + "---|" * (len(SESSIONS) + 1))
    for label in all_labels:
        row = [str(raw_dist_by_session[s].get(label, 0)) for s in SESSIONS]
        lines.append(f"| {label} | " + " | ".join(row) + f" | {raw_dist_overall[label]} |")
    lines.append(f"| **total** | " + " | ".join(str(sum(raw_dist_by_session[s].values())) for s in SESSIONS) + f" | {n_utterances_total} |")
    lines.append("")

    lines.append("## 3. Protocol comparison: 6-class vs. 4-class\n")
    lines.append(f"- **6-class** {{{', '.join(SIX_CLASS_MAP.values())}}} (codes: {{{', '.join(SIX_CLASS_MAP.keys())}}})")
    lines.append(f"- **4-class** {{{', '.join(sorted(set(FOUR_CLASS_MAP.values())))}}} (codes: {{{', '.join(FOUR_CLASS_MAP.keys())}}}, happy+excited merged)\n")

    lines.append("### Per-session utterance counts kept, by protocol\n")
    lines.append("| session | 6-class kept | 4-class kept | total in session |")
    lines.append("|---|---|---|---|")
    for s in SESSIONS:
        total_s = sum(raw_dist_by_session[s].values())
        lines.append(f"| {s} | {six_class['by_session'][s]} | {four_class['by_session'][s]} | {total_s} |")
    lines.append(
        f"| **total** | **{six_class['n_kept']}** | **{four_class['n_kept']}** | **{n_utterances_total}** |"
    )
    lines.append("")
    lines.append(
        f"**Discarded utterances:** 6-class discards **{six_class['n_discarded']}** "
        f"({six_class['n_discarded'] / n_utterances_total:.1%}); "
        f"4-class discards **{four_class['n_discarded']}** ({four_class['n_discarded'] / n_utterances_total:.1%}).\n"
    )

    lines.append("### Session4 (val) / Session5 (test) specifically\n")
    lines.append("| split | session | 6-class kept | 4-class kept | total in session |")
    lines.append("|---|---|---|---|---|")
    for label, s in (("val", "Session4"), ("test", "Session5")):
        total_s = sum(raw_dist_by_session[s].values())
        lines.append(f"| {label} | {s} | {six_class['by_session'][s]} | {four_class['by_session'][s]} | {total_s} |")
    lines.append("")

    lines.append("### Per-class counts (kept utterances only)\n")
    lines.append("**6-class:**\n")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for cls, cnt in sorted(six_class["by_class"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls} | {cnt} |")
    lines.append("\n**4-class:**\n")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for cls, cnt in sorted(four_class["by_class"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls} | {cnt} |")
    lines.append("")

    lines.append("### Literature comparability (one sentence per protocol, no further recommendation)\n")
    lines.append(
        "- **4-class** {angry, happy+excited, sad, neutral} is the more common split in the emotion-"
        "recognition literature (comparable to most published IEMOCAP SOTA tables, which pool "
        "happy+excited and drop frustrated/fear/disgust/surprise/xxx/other).\n"
        "- **6-class** {angry, happy, excited, sad, neutral, frustrated} keeps happy and excited "
        "separate and adds frustrated -- less common as a direct comparison point, but closer to "
        "MELD's own 7-class granularity (docs/PHASE_T.md) if cross-corpus label-set similarity to "
        "this project's MELD work is the priority instead of literature comparability.\n"
    )

    lines.append(
        "## STOP -- protocol choice is not made here\n\n"
        "Per the B1 instructions: counts presented above, no preprocessing attempted, no processed "
        "index built. Waiting on the 4-class vs. 6-class decision before any further N5-B work."
    )

    report_text = "\n".join(lines)
    (PROJECT_ROOT / "docs" / "iemocap_inventory.md").write_text(report_text)
    print(f"[done] wrote docs/iemocap_inventory.md")
    print(f"[done] n_dialogues={n_dialogues_total} n_utterances={n_utterances_total} missing_files={len(missing_files)}")

    summary = {
        "data_root": str(DATA_ROOT),
        "n_dialogues_total": n_dialogues_total,
        "n_utterances_total": n_utterances_total,
        "session_file_presence": session_file_presence,
        "missing_files": missing_files,
        "raw_dist_overall": dict(raw_dist_overall),
        "raw_dist_by_session": {s: dict(c) for s, c in raw_dist_by_session.items()},
        "six_class": six_class,
        "four_class": four_class,
    }
    (PROJECT_ROOT / "outputs" / "iemocap_inventory_data.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote outputs/iemocap_inventory_data.json")


if __name__ == "__main__":
    main()
