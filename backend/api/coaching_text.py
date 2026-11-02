"""Coaching prompts and plain-text cleanup for model tips.

The stroke model only emits labels. The prompt has to name one fault from those
labels, or the model falls back to advice that would fit any shot. Markdown
emphasis is removed here because the UI renders the string as plain text.
"""
import re

_QUALITY_SCORE = {
    "Beginner": 1, "Beginner+": 2, "Developing": 3,
    "Competent": 4, "Competent+": 5, "Proficient": 6,
    "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10,
}
_SKIP = {"", "none", "unknown", "other", "n/a"}
_BOLD = re.compile(r"\*\*([\s\S]+?)\*\*")
_UNDER = re.compile(r"__([\s\S]+?)__")
_LEADING = re.compile(r"^(?:[-•]|\d+[.)])\s*")

_PLAIN = (
    "Plain text only. No markdown, no asterisks, no bold, no bullets, and no numbering."
)
_BAN = (
    "Do not use these phrases unless they are the primary fault: "
    "eye on the shuttle, center of gravity, prepare early, stay balanced, ready position."
)


def plain_coaching_text(text: str) -> str:
    """Turn **bold** and similar markup into the words inside it."""
    if not text:
        return ""
    cleaned = _BOLD.sub(r"\1", text)
    cleaned = _UNDER.sub(r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "").replace("*", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def coaching_lines(text: str, limit: int) -> list[str]:
    lines = []
    for raw in plain_coaching_text(text).splitlines():
        line = _LEADING.sub("", raw.strip()).strip()
        if line:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def quality_score(label, default: int = 5) -> int:
    base = str(label or "").split("(", 1)[0].strip()
    return _QUALITY_SCORE.get(base, default)


def _metric(metrics: dict, key: str, default: str) -> tuple[str, float | None]:
    value = (metrics or {}).get(key, {})
    if isinstance(value, dict):
        label = str(value.get("label") or default)
        conf = value.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            return label, float(conf)
        return label, None
    if value in (None, ""):
        return default, None
    return str(value), None


def _known(label: str) -> bool:
    return str(label or "").strip().lower() not in _SKIP


def _metric_line(title: str, label: str, confidence: float | None) -> str:
    if confidence is None or confidence <= 0:
        return f"{title}: {label}"
    return f"{title}: {label} ({confidence:.0%})"


def primary_fault(stroke: str, confidence: float, metrics: dict, numeric_quality: int) -> str:
    """One fault the tip is allowed to correct, chosen from the labels we have."""
    if confidence < 0.5:
        return (
            "Primary fault: the stroke read is uncertain. "
            "Say the camera angle is unclear and do not prescribe a technique change."
        )
    tech, _ = _metric(metrics, "technique", "Unknown")
    place, _ = _metric(metrics, "placement", "Unknown")
    pos, _ = _metric(metrics, "position", "Unknown")
    intent, _ = _metric(metrics, "intent", "None")
    if numeric_quality <= 4 and _known(tech):
        return f"Primary fault: {tech} on the {stroke}. Quality is {numeric_quality}/10."
    if numeric_quality >= 8 and _known(place):
        where = f" from {pos}" if _known(pos) else ""
        return f"Primary fault: placement {place}{where} on the {stroke}."
    if _known(tech):
        return f"Primary fault: {tech} on the {stroke}."
    if _known(place):
        return f"Primary fault: placement {place} on the {stroke}."
    if _known(pos):
        return f"Primary fault: court position {pos} on the {stroke}."
    if _known(intent):
        return f"Primary fault: intent {intent} on the {stroke}."
    return f"Primary fault: the {stroke}. Quality is {numeric_quality}/10."


def rally_sequence(timeline) -> str:
    labels = []
    for event in timeline or []:
        label = str((event or {}).get("label") or "").strip()
        if not _known(label):
            continue
        if not labels or labels[-1] != label:
            labels.append(label)
    if len(labels) < 2:
        return ""
    return " → ".join(labels[:8])


def _evidence(stroke, subtype, confidence, metrics, quality_label, numeric_quality, timeline) -> str:
    tech, tech_c = _metric(metrics, "technique", "Unknown")
    place, place_c = _metric(metrics, "placement", "Unknown")
    pos, pos_c = _metric(metrics, "position", "Unknown")
    intent, intent_c = _metric(metrics, "intent", "None")
    lines = [
        primary_fault(stroke, confidence, metrics, numeric_quality),
        f"Stroke: {stroke} ({subtype}), confidence {confidence:.0%}",
        _metric_line("Technique", tech, tech_c),
        _metric_line("Placement", place, place_c),
        _metric_line("Court position", pos, pos_c),
        _metric_line("Intent", intent, intent_c),
        f"Quality: {numeric_quality}/10 ({quality_label})",
    ]
    sequence = rally_sequence(timeline)
    if sequence:
        lines.append(f"Rally sequence: {sequence}")
    return "\n".join(lines)


def clip_coaching_prompt(stroke, subtype, confidence, metrics, quality_label, numeric_quality, timeline=None) -> str:
    sequence_rule = ""
    if rally_sequence(timeline):
        sequence_rule = "If a rally sequence is listed, the third tip must address that sequence.\n"
    return (
        "You are a badminton coach. Write exactly 3 tips. Each tip is one sentence.\n"
        "Each tip must name this stroke and correct one label from the evidence. "
        "A tip that would fit any stroke is wrong.\n"
        "The first tip corrects only the primary fault. "
        "The second tip corrects a different label: technique, placement, court position, or intent.\n"
        f"{sequence_rule}"
        f"{_PLAIN}\n"
        f"{_BAN}\n\n"
        f"{_evidence(stroke, subtype, confidence, metrics, quality_label, numeric_quality, timeline)}\n"
    )


def frame_tip_prompt(stroke, subtype, confidence, metrics, quality_label, numeric_quality) -> str:
    return (
        "You are a badminton coach. Reply with one sentence, at most 20 words.\n"
        "Name this stroke and correct only the primary fault. "
        "A tip that would fit any stroke is wrong.\n"
        f"{_PLAIN}\n"
        f"{_BAN}\n\n"
        f"{_evidence(stroke, subtype, confidence, metrics, quality_label, numeric_quality, None)}\n"
    )


def live_cue_worth_saying(result: dict) -> bool:
    """Live audio stays quiet unless this window is a specific stroke."""
    label = str((result or {}).get("label") or "").strip()
    if not _known(label):
        return False
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return confidence >= 0.5


def live_coaching_prompt(result: dict, max_words: int) -> str:
    metrics = result.get("metrics") or {}
    stroke = result.get("label") or "Other"
    confidence = float(result.get("confidence") or 0.0)
    quality = metrics.get("quality") or "Developing"
    subtype, _ = _metric(metrics, "subtype", "None")
    return (
        "Courtside cue during a point. "
        f"One sentence, at most {max_words} words, ending with a period. "
        "Say the stroke and one correction for the primary fault. "
        "No praise, no explanation, no second sentence, no markdown, no asterisks.\n"
        f"{_BAN}\n\n"
        f"{_evidence(stroke, subtype, confidence, metrics, quality, quality_score(quality), None)}\n"
    )
