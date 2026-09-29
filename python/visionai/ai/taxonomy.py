"""Expression taxonomy and the wording used to present model output.

Everything the user interface says about a person comes from this module, so the
"prediction, not a fact" framing is enforced in one place instead of being
repeated across widgets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

ESTIMATE_PREFIX = "Detected facial expression"
DISCLAIMER = (
    "AI-estimated facial expression based on visible facial features. "
    "This is a model's prediction, not a measurement of a person's actual "
    "emotion or mental state."
)


@dataclass(frozen=True)
class EmotionClass:
    """One expression label plus everything needed to render it."""

    name: str
    emoji: str
    color: tuple[int, int, int]
    group: str
    description: str

    @property
    def bgr(self) -> tuple[int, int, int]:
        return (self.color[2], self.color[1], self.color[0])


def _cls(name: str, emoji: str, color: tuple[int, int, int], group: str, desc: str) -> EmotionClass:
    return EmotionClass(name=name, emoji=emoji, color=color, group=group, description=desc)


PRIMARY_CLASSES: tuple[EmotionClass, ...] = (
    _cls("happy", "😊", (76, 217, 96), "positive", "raised cheeks, upturned mouth"),
    _cls("sad", "😢", (66, 165, 245), "negative", "drooping mouth, lowered brow"),
    _cls("angry", "😠", (240, 84, 84), "negative", "lowered brow, tightened mouth"),
    _cls("fear", "😨", (171, 128, 255), "negative", "raised brow, widened eyes"),
    _cls("surprise", "😲", (255, 196, 61), "neutral_positive", "raised brow, open mouth"),
    _cls("disgust", "🤢", (163, 190, 60), "negative", "wrinkled nose, raised upper lip"),
    _cls("neutral", "😐", (176, 190, 197), "neutral", "no strong expression cues"),
)

EXTENDED_CLASSES: tuple[EmotionClass, ...] = (
    _cls("joy", "😄", (255, 193, 7), "positive", "broad smile, raised cheeks"),
    _cls("love", "😍", (233, 30, 99), "positive", "softened gaze, smile"),
    _cls("calm", "😌", (129, 212, 250), "neutral_positive", "relaxed features"),
    _cls("content", "🙂", (200, 230, 128), "positive", "gentle closed-mouth smile"),
    _cls("excited", "🤩", (255, 128, 0), "positive", "wide eyes, broad smile"),
    _cls("confident", "😎", (120, 200, 255), "positive", "level gaze, slight smile"),
    _cls("grateful", "🙏", (255, 204, 128), "positive", "soft smile, softened brow"),
    _cls("affection", "🥰", (240, 98, 146), "positive", "warm gaze, gentle smile"),
    _cls("relief", "😌", (129, 212, 250), "neutral_positive", "released tension"),
    _cls("amusement", "😆", (255, 214, 0), "positive", "open smile, raised cheeks"),
    _cls("disappointed", "😞", (120, 144, 156), "negative", "lowered corners of mouth"),
    _cls("lonely", "😔", (96, 125, 139), "negative", "downturned mouth, lowered gaze"),
    _cls("worried", "😟", (149, 117, 205), "negative", "raised inner brow"),
    _cls("anxious", "😰", (206, 147, 216), "negative", "tight mouth, raised brow"),
    _cls("frustrated", "😤", (230, 74, 25), "negative", "compressed lips"),
    _cls("overwhelmed", "😵", (156, 39, 176), "negative", "wide eyes, open mouth"),
    _cls("grief", "😭", (63, 81, 181), "negative", "furrowed brow, tears"),
    _cls("annoyed", "😒", (117, 107, 92), "negative", "half-lidded gaze"),
    _cls("irritated", "😑", (141, 110, 99), "negative", "narrowed eyes"),
    _cls("rage", "😡", (198, 40, 40), "negative", "strongly lowered brow"),
    _cls("terrified", "😱", (123, 31, 162), "negative", "wide eyes, open mouth"),
    _cls("distressed", "😫", (93, 64, 55), "negative", "scrunched features"),
    _cls("exhausted", "😩", (93, 64, 55), "negative", "drooping eyelids"),
    _cls("confused", "😕", (144, 164, 174), "neutral", "asymmetric brow"),
    _cls("contempt", "😏", (141, 110, 49), "negative", "asymmetric mouth corner"),
    _cls("embarrassed", "😳", (244, 143, 177), "negative", "flushed, averted gaze"),
    _cls("nervous", "😬", (188, 140, 255), "negative", "tightened mouth"),
    _cls("uncomfortable", "🫣", (160, 140, 130), "negative", "tightened lower face"),
    _cls("ashamed", "😶", (121, 85, 72), "negative", "lowered gaze, pressed lips"),
    _cls("concerned", "🫤", (94, 143, 199), "neutral_positive", "slightly raised brow"),
    _cls("thinking", "🤔", (144, 202, 249), "neutral", "hand near chin, side gaze"),
    _cls("skeptical", "🤨", (149, 117, 205), "neutral", "one raised brow"),
    _cls("curious", "🧐", (129, 199, 212), "neutral_positive", "raised brow, tilted head"),
    _cls("sarcastic", "😏", (141, 110, 49), "neutral", "asymmetric smile"),
    _cls("relaxed", "😌", (178, 223, 219), "neutral_positive", "loose jaw"),
    _cls("bored", "🥱", (144, 164, 174), "neutral", "half-lidded eyes"),
    _cls("sleepy", "😴", (63, 81, 188), "neutral", "closed eyes, slack mouth"),
    _cls("tired", "🥱", (121, 134, 203), "neutral", "drooping features"),
    _cls("shocked", "😱", (244, 67, 54), "neutral_positive", "widened eyes"),
    _cls("playful", "😜", (255, 179, 0), "positive", "asymmetric smile"),
)

ALLOWED_GROUPS: tuple[str, ...] = ("positive", "neutral", "neutral_positive", "negative")

_BY_NAME: dict[str, EmotionClass] = {c.name: c for c in PRIMARY_CLASSES + EXTENDED_CLASSES}

#: Public aliases so common dataset label spellings resolve to canonical names.
LABEL_ALIASES: dict[str, str] = {
    "happiness": "happy",
    "joy": "joy",
    "anger": "angry",
    "angriness": "angry",
    "mad": "angry",
    "sadness": "sad",
    "sorrow": "sad",
    "fearful": "fear",
    "surprised": "surprise",
    "surprize": "surprise",
    "disgusted": "disgust",
    "normal": "neutral",
    "calm_neutral": "neutral",
    "meh": "neutral",
    "none": "neutral",
}

DEFAULT_CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in PRIMARY_CLASSES)


def all_classes() -> tuple[EmotionClass, ...]:
    return PRIMARY_CLASSES + EXTENDED_CLASSES


def get(name: str) -> EmotionClass | None:
    key = str(name).strip().lower()
    return _BY_NAME.get(LABEL_ALIASES.get(key, key))


def resolve(names: Iterable[str] | None = None) -> tuple[EmotionClass, ...]:
    """Resolve a set of class names to taxonomy entries.

    ``None`` or an empty selection yields the seven primary classes. Unknown
    names are dropped so a mis-typed extra class cannot break rendering.
    """
    if not names:
        return PRIMARY_CLASSES
    resolved: list[EmotionClass] = []
    seen: set[str] = set()
    for name in names:
        entry = get(name)
        if entry is not None and entry.name not in seen:
            seen.add(entry.name)
            resolved.append(entry)
    return tuple(resolved) or PRIMARY_CLASSES


def class_names(classes: Iterable[EmotionClass] | None = None) -> tuple[str, ...]:
    return tuple(c.name for c in (classes or PRIMARY_CLASSES))


def normalize_label(raw: str, allowed: Iterable[str] | None = None) -> str | None:
    """Map a raw model label onto one of ``allowed`` class names."""
    if allowed is not None:
        allowed_set = {str(a).strip().lower() for a in allowed}
        if not allowed_set:
            return None
        entry = get(raw)
        if entry is not None and entry.name in allowed_set:
            return entry.name
        key = str(raw).strip().lower()
        if key in allowed_set:
            return key
        alias = LABEL_ALIASES.get(key)
        if alias is not None and alias in allowed_set:
            return alias
        return None
    entry = get(raw)
    return entry.name if entry is not None else str(raw).strip().lower() or None


def align_scores(
    scores: Mapping[str, float] | Iterable[float], labels: Iterable[str]
) -> dict[str, float]:
    """Produce a normalised score mapping keyed by canonical class names.

    Accepts either a label->score mapping or a positional sequence aligned with
    ``labels``. Folds aliases (for example ``anger`` into ``angry``) and rescales
    so the values sum to 1.0.
    """
    label_list = [str(label) for label in labels]
    if isinstance(scores, Mapping):
        raw: dict[str, float] = {}
        for key, value in scores.items():
            canonical = normalize_label(key, label_list) or str(key).strip().lower()
            raw[canonical] = raw.get(canonical, 0.0) + float(value)
    else:
        values = list(scores)
        raw = {label: float(values[i]) if i < len(values) else 0.0 for i, label in enumerate(label_list)}

    total = sum(v for v in raw.values() if v > 0)
    if total <= 0:
        count = max(1, len(label_list))
        return dict.fromkeys(label_list, 1.0 / count)
    return {label: max(0.0, value) / total for label, value in raw.items()}


def color_for(name: str) -> tuple[int, int, int]:
    entry = get(name)
    return entry.color if entry is not None else (176, 190, 197)


def bgr_for(name: str) -> tuple[int, int, int]:
    entry = get(name)
    return entry.bgr if entry is not None else (197, 190, 176)


def emoji_for(name: str) -> str:
    entry = get(name)
    return entry.emoji if entry is not None else "❔"


def title(name: str) -> str:
    return str(name or "unknown").replace("_", " ").strip().title()


def describe(name: str, confidence: float | None = None) -> str:
    """The canonical user-facing sentence for a single prediction."""
    text = f"{ESTIMATE_PREFIX}: {title(name)}"
    if confidence is not None:
        text += f" ({float(confidence) * 100:.0f}% confidence)"
    return text


def describe_track(track_id: int, name: str, confidence: float | None = None) -> str:
    text = f"ID {int(track_id):02d} → {title(name)}"
    if confidence is not None:
        text += f"  {float(confidence) * 100:.0f}%"
    return text


def banner() -> str:
    return f"{ESTIMATE_PREFIX} (AI estimate). {DISCLAIMER}"
