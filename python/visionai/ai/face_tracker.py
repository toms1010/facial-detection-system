"""Multi-face tracking by IoU association with per-track label smoothing.

The tracker is intentionally simple and dependency-free: greedy highest-IoU
matching, exponential box smoothing, and a decaying score for a label that has
not been re-observed recently. That is enough to keep a stable ``ID 01`` on a
face across frames, which is what the UI needs.

It is not a full multi-object tracker: there is no motion model, so a face that
disappears and reappears elsewhere gets a new ID. That trade-off keeps latency
well under a millisecond per frame.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from visionai.ai.models.base import Box, ExpressionResult, FaceDetection
from visionai.ai.taxonomy import align_scores as taxonomy_align
from visionai.ai.taxonomy import describe_track

LOG = logging.getLogger(__name__)

_ID_COUNTER = itertools.count(1)


def reset_id_counter(start: int = 1) -> None:
    """Test hook so track IDs are deterministic across runs."""
    global _ID_COUNTER
    _ID_COUNTER = itertools.count(start)


@dataclass
class TrackedFace:
    """A face with a stable identity across frames."""

    track_id: int
    box: Box
    detection_score: float
    expression: ExpressionResult | None = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    hits: int = 1
    misses: int = 0
    history: list[Box] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.expression.label if self.expression else "unknown"

    @property
    def confidence(self) -> float:
        return self.expression.confidence if self.expression else 0.0

    @property
    def is_stale(self) -> bool:
        return self.misses > 0

    def smoothed_scores(self) -> dict[str, float]:
        return dict(self.expression.scores) if self.expression else {}

    def age_frames_estimate(self, inference_fps: float) -> float:
        return self.hits / max(1e-3, inference_fps)


class FaceTracker:
    """Associates detections with persistent track identities."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        smooth: bool = True,
        smooth_alpha: float = 0.45,
        hold_labels: int = 12,
    ) -> None:
        self.iou_threshold = float(max(0.0, min(1.0, iou_threshold)))
        self.max_age = int(max(0, max_age))
        self.smooth = bool(smooth)
        self.smooth_alpha = float(max(0.0, min(1.0, smooth_alpha)))
        self.hold_labels = int(max(0, hold_labels))
        self._tracks: dict[int, TrackedFace] = {}

    @property
    def tracks(self) -> list[TrackedFace]:
        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        detections: Sequence[FaceDetection],
        expressions: dict[int, ExpressionResult] | None = None,
    ) -> list[TrackedFace]:
        """Match ``detections`` to tracks and return the surviving track list.

        ``expressions`` maps the *index* of each detection to its classification.
        """
        expressions = expressions or {}
        matches, matched_tracks, matched_detections = self._associate(
            [d.box for d in detections]
        )

        for track_id, det_index in matches:
            track = self._tracks[track_id]
            detection = detections[det_index]
            track.box = self._smooth_box(track.box, detection.box)
            track.detection_score = detection.score
            track.hits += 1
            track.misses = 0
            track.last_seen = time.time()
            track.history.append(track.box.copy())
            if len(track.history) > 64:
                del track.history[:-64]
            expression = expressions.get(det_index)
            if expression is not None:
                self._apply_expression(track, expression)

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.misses += 1

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue
            self._spawn(detection, expressions.get(det_index))

        self._prune()
        return self.tracks

    def _associate(
        self, boxes: Sequence[Box]
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Greedy highest-IoU matching between existing tracks and detections.

        Returns the accepted ``(track_id, detection_index)`` pairs, the set of
        matched track IDs and the set of matched detection indices.
        """
        pairs: list[tuple[float, int, int]] = []
        for track_id in sorted(self._tracks):
            for det_index, box in enumerate(boxes):
                iou = self._tracks[track_id].box.iou(box)
                if iou >= self.iou_threshold:
                    pairs.append((iou, track_id, det_index))
        pairs.sort(key=lambda item: item[0], reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, track_id, det_index in pairs:
            if track_id in used_tracks or det_index in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_index)
            matches.append((track_id, det_index))
        return matches, used_tracks, used_dets

    def _smooth_box(self, previous: Box, current: Box) -> Box:
        if not self.smooth:
            return current.copy()
        a = self.smooth_alpha
        return Box(
            previous.x1 + (current.x1 - previous.x1) * a,
            previous.y1 + (current.y1 - previous.y1) * a,
            previous.x2 + (current.x2 - previous.x2) * a,
            previous.y2 + (current.y2 - previous.y2) * a,
        )

    def _apply_expression(self, track: TrackedFace, expression: ExpressionResult) -> None:
        if self.smooth and track.expression is not None and track.expression.scores:
            alpha = self.smooth_alpha
            keys = set(track.expression.scores) | set(expression.scores)
            blended = {
                key: (1.0 - alpha) * track.expression.scores.get(key, 0.0)
                + alpha * expression.scores.get(key, 0.0)
                for key in keys
            }
            scores = taxonomy_align(blended, keys)
            label, confidence = max(scores.items(), key=lambda kv: kv[1])
            expression = ExpressionResult(
                label=label,
                confidence=confidence,
                scores=scores,
                model=expression.model,
                is_heuristic=expression.is_heuristic,
                is_confident=expression.is_confident,
                latency_ms=expression.latency_ms,
                alternatives=expression.alternatives,
            )
        expression.box = track.box
        expression.track_id = track.track_id
        track.expression = expression

    def _spawn(self, detection: FaceDetection, expression: ExpressionResult | None) -> TrackedFace:
        track_id = next(_ID_COUNTER)
        track = TrackedFace(
            track_id=track_id,
            box=detection.box.copy(),
            detection_score=detection.score,
            expression=expression,
        )
        if expression is not None:
            expression.box = track.box
            expression.track_id = track_id
        self._tracks[track_id] = track
        return track

    def _prune(self) -> None:
        stale = [tid for tid, t in self._tracks.items() if t.misses > self.max_age]
        for tid in stale:
            del self._tracks[tid]

    def describe(self) -> dict:
        return {
            "tracks": self.active_count,
            "iou_threshold": self.iou_threshold,
            "max_age": self.max_age,
            "smooth": self.smooth,
        }


def summarise(tracks: Iterable[TrackedFace]) -> str:
    """Render tracks as the ``ID 01 -> Happy 91%`` lines the UI shows."""
    return "\n".join(
        describe_track(track.track_id, track.label, track.confidence) for track in tracks
    )
