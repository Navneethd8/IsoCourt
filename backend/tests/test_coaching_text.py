from api.coaching_text import (
    clip_coaching_prompt,
    coaching_lines,
    live_coaching_prompt,
    live_cue_worth_saying,
    plain_coaching_text,
    primary_fault,
)


def test_plain_coaching_text_drops_bold_markers():
    assert plain_coaching_text("**Snap** your wrist at contact.") == "Snap your wrist at contact."
    assert plain_coaching_text("Meet the shuttle **higher**.") == "Meet the shuttle higher."
    assert "*" not in plain_coaching_text("Use a **forehand** clear.")


def test_coaching_lines_strip_markup_and_bullets():
    raw = "1. **Extend** the elbow on the smash.\n- Keep a **tight** grip on the drop.\n* third"
    assert coaching_lines(raw, 3) == [
        "Extend the elbow on the smash.",
        "Keep a tight grip on the drop.",
        "third",
    ]


def test_primary_fault_follows_the_read():
    metrics = {
        "technique": {"label": "Forehand", "confidence": 0.9},
        "placement": {"label": "Cross Court", "confidence": 0.7},
        "position": {"label": "Rear Court", "confidence": 0.8},
        "intent": {"label": "Attack", "confidence": 0.6},
    }
    low = primary_fault("Smash", 0.3, metrics, 3)
    assert "uncertain" in low
    weak = primary_fault("Smash", 0.82, metrics, 3)
    assert "Forehand" in weak
    sharp = primary_fault("Smash", 0.82, metrics, 9)
    assert "Cross Court" in sharp


def test_clip_prompt_names_the_fault_and_forbids_markdown():
    metrics = {"technique": {"label": "Forehand", "confidence": 0.91}}
    timeline = [{"label": "Clear"}, {"label": "Smash"}, {"label": "Drop"}]
    prompt = clip_coaching_prompt("Smash", "Jump", 0.82, metrics, "Developing", 3, timeline)
    assert "Primary fault: Forehand on the Smash" in prompt
    assert "Rally sequence: Clear → Smash → Drop" in prompt
    assert "No markdown, no asterisks" in prompt
    assert "eye on the shuttle" in prompt


def test_live_prompt_is_a_short_spoken_cue():
    result = {
        "label": "Drop",
        "confidence": 0.77,
        "metrics": {
            "subtype": {"label": "Slow"},
            "technique": {"label": "Backhand"},
            "placement": {"label": "Net"},
            "quality": "Advanced",
        },
    }
    prompt = live_coaching_prompt(result, 8)
    assert "at most 8 words" in prompt
    assert "No praise" in prompt
    assert "no asterisks" in prompt
    assert "Backhand" in prompt


def test_live_cue_stays_quiet_when_the_read_is_weak():
    assert live_cue_worth_saying({"label": "Smash", "confidence": 0.8}) is True
    assert live_cue_worth_saying({"label": "Other", "confidence": 0.9}) is False
    assert live_cue_worth_saying({"label": "Smash", "confidence": 0.4}) is False
