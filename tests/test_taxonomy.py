"""The expression taxonomy and the wording shown to the user."""

from __future__ import annotations

import pytest

from visionai.ai import taxonomy


def test_seven_primary_classes() -> None:
    assert [c.name for c in taxonomy.PRIMARY_CLASSES] == [
        "happy",
        "sad",
        "angry",
        "fear",
        "surprise",
        "disgust",
        "neutral",
    ]


def test_every_extended_class_is_unique() -> None:
    names = [c.name for c in taxonomy.all_classes()]
    assert len(names) == len(set(names))


def test_extended_classes_have_emoji_colour_and_description() -> None:
    for entry in taxonomy.EXTENDED_CLASSES:
        assert entry.emoji and entry.emoji != "❔", entry.name
        assert entry.description, entry.name
        assert all(0 <= channel <= 255 for channel in entry.color), entry.name


def test_every_extended_class_is_in_the_documented_roadmap() -> None:
    expected = {
        "joy", "love", "calm", "content", "excited", "confident", "grateful", "affection",
        "relief", "amusement", "disappointed", "lonely", "worried", "anxious", "frustrated",
        "overwhelmed", "grief", "annoyed", "irritated", "rage", "terrified", "distressed",
        "exhausted", "confused", "contempt", "embarrassed", "nervous", "uncomfortable",
        "ashamed", "concerned", "thinking", "skeptical", "curious", "sarcastic", "relaxed",
        "bored", "sleepy", "tired", "shocked", "playful",
    }
    assert {c.name for c in taxonomy.EXTENDED_CLASSES} == expected


def test_resolve_defaults_to_primary() -> None:
    assert taxonomy.resolve() == taxonomy.PRIMARY_CLASSES
    assert taxonomy.resolve([]) == taxonomy.PRIMARY_CLASSES
    assert taxonomy.resolve(None) == taxonomy.PRIMARY_CLASSES


def test_resolve_drops_unknown_names() -> None:
    resolved = taxonomy.resolve(["happy", "not-an-emotion", "SAD"])
    assert [c.name for c in resolved] == ["happy", "sad"]


def test_aliases_map_to_canonical_names() -> None:
    assert taxonomy.normalize_label("Anger") == "angry"
    assert taxonomy.normalize_label("happiness") == "happy"
    assert taxonomy.normalize_label("Fearful") == "fear"
    assert taxonomy.normalize_label("surprised") == "surprise"


def test_normalize_label_respects_allowed_set() -> None:
    assert taxonomy.normalize_label("anger", ["happy", "angry"]) == "angry"
    assert taxonomy.normalize_label("contempt", ["happy", "angry"]) is None


def test_align_scores_normalises_to_one() -> None:
    aligned = taxonomy.align_scores({"happy": 2.0, "sad": 1.0, "neutral": 1.0}, ["happy", "sad", "neutral"])
    assert sum(aligned.values()) == pytest.approx(1.0)
    assert aligned["happy"] == pytest.approx(0.5)


def test_align_scores_folds_aliases() -> None:
    aligned = taxonomy.align_scores({"anger": 3.0, "happy": 1.0}, ["angry", "happy"])
    assert aligned["angry"] == pytest.approx(0.75)


def test_align_scores_handles_zero_total() -> None:
    aligned = taxonomy.align_scores({"happy": 0.0, "sad": 0.0}, ["happy", "sad"])
    assert aligned["happy"] == pytest.approx(0.5)


def test_align_scores_accepts_a_positional_sequence() -> None:
    aligned = taxonomy.align_scores([1.0, 3.0], ["happy", "sad"])
    assert aligned["sad"] == pytest.approx(0.75)


def test_align_scores_coerces_non_string_keys() -> None:
    """A model may hand back an enum or index key; it must not blow up."""
    aligned = taxonomy.align_scores(
        {taxonomy.get("happy"): 3.0, taxonomy.get("sad"): 1.0},
        ["happy", "sad"],
    )
    assert aligned["happy"] == pytest.approx(0.75)
    assert aligned["sad"] == pytest.approx(0.25)
    assert all(isinstance(key, str) for key in aligned)


def test_align_scores_output_keys_are_always_strings() -> None:
    aligned = taxonomy.align_scores({0: 1.0, 1: 3.0}, ["happy", "sad"])
    assert all(isinstance(key, str) for key in aligned)
    assert sum(aligned.values()) == pytest.approx(1.0)


def test_align_scores_accepts_emotion_class_labels() -> None:
    aligned = taxonomy.align_scores(
        [3.0, 1.0], [taxonomy.get("happy"), taxonomy.get("sad")]
    )
    assert aligned == {"happy": pytest.approx(0.75), "sad": pytest.approx(0.25)}


def test_label_text_unwraps_entries_and_named_objects() -> None:
    entry = taxonomy.get("happy")
    assert entry is not None
    assert taxonomy.label_text(entry) == "happy"
    assert taxonomy.label_text(" Happy ") == " Happy "
    assert taxonomy.label_text(7) == "7"

    class Named:
        name = "happy"

    assert taxonomy.label_text(Named()) == "happy"


def test_normalize_label_accepts_an_emotion_class() -> None:
    entry = taxonomy.get("happy")
    assert entry is not None
    assert taxonomy.normalize_label(entry, ["happy", "sad"]) == "happy"
    assert taxonomy.normalize_label(entry) == "happy"


def test_colour_helpers_are_bgr_swapped() -> None:
    entry = taxonomy.get("happy")
    assert entry is not None
    assert taxonomy.bgr_for("happy") == (entry.color[2], entry.color[1], entry.color[0])


def test_unknown_label_falls_back_safely() -> None:
    assert taxonomy.get("nonsense") is None
    assert taxonomy.color_for("nonsense") == (176, 190, 197)
    assert taxonomy.emoji_for("nonsense") == "❔"


def test_describe_uses_estimate_wording() -> None:
    text = taxonomy.describe("happy", 0.91)
    assert text == "Detected facial expression: Happy (91% confidence)"


def test_describe_track_uses_id_format() -> None:
    assert taxonomy.describe_track(1, "happy", 0.91) == "ID 01 → Happy  91%"
    assert taxonomy.describe_track(12, "neutral") == "ID 12 → Neutral"


def test_disclaimer_refuses_to_claim_mental_state() -> None:
    disclaimer = taxonomy.DISCLAIMER.lower()
    assert "not a measurement" in disclaimer
    assert "actual" in disclaimer


def test_banner_contains_the_prefix_and_disclaimer() -> None:
    banner = taxonomy.banner()
    assert taxonomy.ESTIMATE_PREFIX in banner
    assert "prediction" in banner.lower()


def test_title_is_human_readable() -> None:
    assert taxonomy.title("no_entry") == "No Entry"
