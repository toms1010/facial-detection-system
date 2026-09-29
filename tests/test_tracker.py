"""Multi-face tracking: identity stability, ageing and label smoothing."""

from __future__ import annotations

import pytest

from visionai.ai.face_tracker import FaceTracker, reset_id_counter, summarise
from visionai.ai.models.base import Box, ExpressionResult, FaceDetection


@pytest.fixture(autouse=True)
def deterministic_ids() -> None:
    reset_id_counter(1)


def detection(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> FaceDetection:
    return FaceDetection(box=Box(x1, y1, x2, y2), score=score)


def expression(label: str = "happy", confidence: float = 0.8) -> ExpressionResult:
    return ExpressionResult.from_scores(
        {label: confidence, "neutral": 1 - confidence}, model="test"
    )


class TestSingleFace:
    def test_first_detection_creates_a_track(self) -> None:
        tracker = FaceTracker()
        tracks = tracker.update([detection(0, 0, 50, 50)], {0: expression()})
        assert len(tracks) == 1
        assert tracks[0].label == "happy"

    def test_identity_is_stable_across_frames(self) -> None:
        tracker = FaceTracker()
        first = tracker.update([detection(0, 0, 50, 50)], {0: expression()})[0]
        for _ in range(5):
            tracks = tracker.update([detection(2, 2, 52, 52)], {0: expression()})
        assert len(tracks) == 1
        assert tracks[0].track_id == first.track_id
        assert tracks[0].hits == 6

    def test_track_ages_when_it_disappears(self) -> None:
        tracker = FaceTracker(max_age=2)
        tracker.update([detection(0, 0, 50, 50)], {})
        tracker.update([], {})
        assert tracker.active_count == 1
        tracker.update([], {})
        assert tracker.active_count == 1
        tracker.update([], {})
        assert tracker.active_count == 0

    def test_box_is_smoothed_towards_the_detection(self) -> None:
        tracker = FaceTracker(smooth=True, smooth_alpha=0.5, iou_threshold=0.1)
        tracker.update([detection(0, 0, 100, 100)], {})
        tracks = tracker.update([detection(20, 20, 120, 120)], {})
        assert len(tracks) == 1
        assert (tracks[0].box.x1, tracks[0].box.x2) == (10.0, 110.0)

    def test_smoothing_can_be_disabled(self) -> None:
        tracker = FaceTracker(smooth=False, iou_threshold=0.1)
        tracker.update([detection(0, 0, 100, 100)], {})
        tracks = tracker.update([detection(20, 20, 120, 120)], {})
        assert tracks[0].box.x1 == 20

    def test_a_large_jump_loses_the_identity(self) -> None:
        tracker = FaceTracker(smooth=True, smooth_alpha=0.5)
        first = tracker.update([detection(0, 0, 100, 100)], {})[0]
        tracks = tracker.update([detection(400, 400, 500, 500)], {})
        assert {t.track_id for t in tracks} == {first.track_id, first.track_id + 1}


class TestMultipleFaces:
    def test_two_faces_get_two_ids(self) -> None:
        tracker = FaceTracker()
        tracks = tracker.update(
            [detection(0, 0, 50, 50), detection(200, 0, 250, 50)],
            {0: expression("happy"), 1: expression("sad")},
        )
        assert len(tracks) == 2
        assert {t.track_id for t in tracks} == {1, 2}
        assert {t.label for t in tracks} == {"happy", "sad"}

    def test_ids_survive_frame_to_frame(self) -> None:
        tracker = FaceTracker()
        before = {t.track_id: t.label for t in tracker.update(
            [detection(0, 0, 50, 50), detection(200, 0, 250, 50)],
            {0: expression("happy"), 1: expression("sad")},
        )}
        after = {t.track_id: t.label for t in tracker.update(
            [detection(3, 0, 53, 50), detection(198, 0, 248, 50)],
            {0: expression("happy"), 1: expression("sad")},
        )}
        assert before == after

    def test_crossing_faces_keep_distinct_identities(self) -> None:
        tracker = FaceTracker(iou_threshold=0.2, smooth=False)
        tracker.update(
            [detection(0, 0, 50, 50), detection(100, 0, 150, 50)],
            {0: expression("happy"), 1: expression("sad")},
        )
        tracker.update(
            [detection(20, 0, 70, 50), detection(120, 0, 170, 50)],
            {0: expression("happy"), 1: expression("sad")},
        )
        tracks = tracker.tracks
        assert len(tracks) == 2
        happy = [t for t in tracks if t.label == "happy"]
        sad = [t for t in tracks if t.label == "sad"]
        assert len(happy) == 1 and len(sad) == 1
        assert happy[0].track_id != sad[0].track_id

    def test_a_new_face_gets_a_new_id(self) -> None:
        tracker = FaceTracker()
        tracker.update([detection(0, 0, 50, 50)], {0: expression()})
        tracks = tracker.update(
            [detection(0, 0, 50, 50), detection(300, 0, 350, 50)],
            {0: expression(), 1: expression()},
        )
        assert len(tracks) == 2
        assert tracks[1].track_id == 2

    def test_all_detections_are_claimed(self) -> None:
        tracker = FaceTracker(iou_threshold=0.1)
        tracks = tracker.update(
            [detection(0, 0, 50, 50), detection(100, 0, 150, 50), detection(200, 0, 250, 50)],
            {0: expression(), 1: expression(), 2: expression()},
        )
        assert len(tracks) == 3

    def test_summarise_uses_id_format(self) -> None:
        tracker = FaceTracker()
        tracker.update([detection(0, 0, 50, 50)], {0: expression("happy", 0.91)})
        assert summarise(tracker.tracks) == "ID 01 → Happy  91%"


class TestLabelSmoothing:
    def test_label_follows_the_new_observation(self) -> None:
        tracker = FaceTracker(smooth=False)
        tracker.update([detection(0, 0, 50, 50)], {0: expression("happy", 0.99)})
        tracks = tracker.update([detection(0, 0, 50, 50)], {0: expression("sad", 0.99)})
        assert tracks[0].label == "sad"

    def test_smoothing_reduces_jitter(self) -> None:
        tracker = FaceTracker(smooth=True, smooth_alpha=0.3)
        tracker.update([detection(0, 0, 50, 50)], {0: expression("happy", 1.0)})
        tracks = tracker.update([detection(0, 0, 50, 50)], {0: expression("sad", 1.0)})
        assert tracks[0].confidence < 1.0

    def test_expression_carries_the_track_id(self) -> None:
        tracker = FaceTracker()
        tracks = tracker.update([detection(0, 0, 50, 50)], {0: expression()})
        assert tracks[0].expression is not None
        assert tracks[0].expression.track_id == tracks[0].track_id

    def test_missing_expression_keeps_the_previous_label(self) -> None:
        tracker = FaceTracker()
        tracker.update([detection(0, 0, 50, 50)], {0: expression("happy")})
        tracks = tracker.update([detection(0, 0, 50, 50)], {})
        assert tracks[0].label == "happy"


class TestEdgeCases:
    def test_empty_detection_list(self) -> None:
        tracker = FaceTracker()
        assert tracker.update([]) == []

    def test_reset_clears_everything(self) -> None:
        tracker = FaceTracker()
        tracker.update([detection(0, 0, 50, 50)], {})
        tracker.reset()
        assert tracker.active_count == 0

    def test_zero_max_age_drops_immediately(self) -> None:
        tracker = FaceTracker(max_age=0)
        tracker.update([detection(0, 0, 50, 50)], {})
        assert tracker.update([]) == []

    def test_zero_iou_threshold_matches_everything(self) -> None:
        tracker = FaceTracker(iou_threshold=0.0)
        tracker.update([detection(0, 0, 50, 50)], {})
        tracks = tracker.update([detection(0, 0, 50, 50)], {})
        assert len(tracks) == 1

    def test_tracks_are_returned_in_id_order(self) -> None:
        tracker = FaceTracker()
        tracker.update(
            [detection(200, 0, 250, 50), detection(0, 0, 50, 50), detection(100, 0, 150, 50)],
            {0: expression(), 1: expression(), 2: expression()},
        )
        assert [t.track_id for t in tracker.tracks] == sorted(t.track_id for t in tracker.tracks)

    def test_history_is_bounded(self) -> None:
        tracker = FaceTracker()
        tracker.update([detection(0, 0, 50, 50)], {})
        for _ in range(80):
            tracker.update([detection(1, 1, 51, 51)], {})
        assert len(tracker.tracks[0].history) <= 64

    def test_describe(self) -> None:
        info = FaceTracker(iou_threshold=0.4, max_age=7).describe()
        assert info["iou_threshold"] == 0.4
        assert info["max_age"] == 7

    def test_tracked_face_without_expression(self) -> None:
        tracker = FaceTracker()
        track = tracker.update([detection(0, 0, 50, 50)], {})[0]
        assert track.label == "unknown"
        assert track.confidence == 0.0
        assert track.expression is None
