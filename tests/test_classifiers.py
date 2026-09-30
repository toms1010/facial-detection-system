"""Expression classification: the heuristic, confidence policy and descriptors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from visionai.ai.errors import FrameError, ModelMissingError
from visionai.ai.models.base import ExpressionResult
from visionai.ai.models.descriptor import (
    ModelDescriptor,
    load_descriptor,
    parse_descriptor,
    preprocess,
)
from visionai.ai.models.heuristic_classifier import (
    CONFIDENCE_CEILING,
    HeuristicExpressionClassifier,
)
from visionai.ai.models.registry import ModelRegistry


class TestHeuristicClassifier:
    def test_is_marked_untrained(self) -> None:
        classifier = HeuristicExpressionClassifier()
        assert classifier.is_heuristic is True
        assert classifier.is_trained_model is False
        assert classifier.requires_weights is False
        assert "untrained" in classifier.describe()["warning"].lower()

    def test_confidence_ceiling_is_below_certainty(self) -> None:
        assert 0 < HeuristicExpressionClassifier.confidence_ceiling < 0.7

    def test_predicts_one_of_the_primary_classes(self, face_crop: np.ndarray) -> None:
        result = HeuristicExpressionClassifier().predict(face_crop)
        assert result.label in {
            "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"
        }

    def test_scores_form_a_distribution(self, face_crop: np.ndarray) -> None:
        result = HeuristicExpressionClassifier().predict(face_crop)
        assert len(result.scores) == 7
        assert all(0.0 <= v <= 1.0 for v in result.scores.values())

    def test_confidence_never_exceeds_the_ceiling(self, face_crop: np.ndarray) -> None:
        result = HeuristicExpressionClassifier().predict(face_crop)
        assert result.confidence <= CONFIDENCE_CEILING + 1e-9

    def test_never_reports_confidence(self, face_crop: np.ndarray) -> None:
        assert HeuristicExpressionClassifier().predict(face_crop).is_confident is False

    def test_is_deterministic(self, face_crop: np.ndarray) -> None:
        first = HeuristicExpressionClassifier().predict(face_crop)
        second = HeuristicExpressionClassifier().predict(face_crop)
        assert first.label == second.label
        assert first.scores == second.scores

    def test_label_matches_the_argmax(self, face_crop: np.ndarray) -> None:
        result = HeuristicExpressionClassifier().predict(face_crop)
        assert max(result.scores, key=lambda label: result.scores[label]) == result.label

    def test_accepts_grayscale_crops(self, face_crop: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        assert HeuristicExpressionClassifier().predict(gray).label

    def test_rejects_empty_and_tiny_crops(self) -> None:
        classifier = HeuristicExpressionClassifier()
        with pytest.raises(FrameError):
            classifier.predict(np.array([]))
        with pytest.raises(FrameError):
            classifier.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_features_are_bounded(self, face_crop: np.ndarray) -> None:
        features = HeuristicExpressionClassifier().extract_features(face_crop)
        for key, value in features.items():
            assert 0.0 <= value <= 1.0, key

    def test_all_primary_classes_receive_a_score(self, face_crop: np.ndarray) -> None:
        scores = HeuristicExpressionClassifier().score(
            HeuristicExpressionClassifier().extract_features(face_crop)
        )
        assert set(scores) == {
            "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"
        }

    def test_temporal_smoothing_changes_the_result(self, frame: np.ndarray) -> None:
        classifier = HeuristicExpressionClassifier(smooth=0.9)
        first = classifier.predict(frame)
        second = classifier.predict(np.zeros_like(frame))
        assert first.label != second.label or first.scores != second.scores

    def test_context_manager(self) -> None:
        with HeuristicExpressionClassifier() as classifier:
            assert classifier.is_ready
        assert not classifier.is_ready

    def test_describe_exposes_wording(self) -> None:
        info = HeuristicExpressionClassifier().describe()
        assert info["is_heuristic"] is True
        assert info["is_trained_model"] is False
        assert info["confidence_ceiling"] == CONFIDENCE_CEILING


class TestExpressionResult:
    def test_from_scores_picks_the_maximum(self) -> None:
        result = ExpressionResult.from_scores(
            {"happy": 0.7, "sad": 0.2, "neutral": 0.1}, model="test"
        )
        assert result.label == "happy"
        assert result.confidence == pytest.approx(0.7)
        assert result.is_confident is True

    def test_threshold_gates_confidence(self) -> None:
        result = ExpressionResult.from_scores(
            {"happy": 0.3, "sad": 0.7}, model="test", threshold=0.9
        )
        assert result.is_confident is False

    def test_heuristic_never_passes_the_threshold(self) -> None:
        result = ExpressionResult.from_scores(
            {"happy": 1.0}, model="test", is_heuristic=True, threshold=0.1
        )
        assert result.is_confident is False

    def test_empty_scores_raise(self) -> None:
        with pytest.raises(Exception):
            ExpressionResult.from_scores({}, model="test")

    def test_label_is_normalised(self) -> None:
        assert ExpressionResult(label="Anger", confidence=0.5).label == "angry"

    def test_confidence_is_clamped(self) -> None:
        assert ExpressionResult(label="happy", confidence=9.0).confidence == 1.0

    def test_top_k_is_sorted(self) -> None:
        result = ExpressionResult.from_scores(
            {"happy": 0.2, "sad": 0.5, "neutral": 0.3}, model="test"
        )
        assert [name for name, _ in result.top_k(2)] == ["sad", "neutral"]

    def test_display_helpers(self) -> None:
        result = ExpressionResult(label="happy", confidence=0.91)
        assert result.display_label == "Happy"
        assert result.emoji == "😊"
        assert result.color == (76, 217, 96)
        assert result.describe() == "Detected facial expression: Happy (91% confidence)"


class TestDescriptor:
    def test_parses_a_full_descriptor(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.onnx"
        weights.write_bytes(b"stub")
        descriptor = parse_descriptor(
            {
                "name": "affectnet",
                "framework": "onnx",
                "classes": ["neutral", "happiness", "anger"],
                "input_size": [112, 112],
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "scale": 1.0,
                "license": "CC-BY-4.0",
            },
            weights,
        )
        assert descriptor.name == "affectnet"
        assert descriptor.input_size == (112, 112)
        assert descriptor.canonical_classes == ("neutral", "happy", "angry")
        assert descriptor.license == "CC-BY-4.0"

    def test_rejects_unknown_framework(self, tmp_path: Path) -> None:
        with pytest.raises(ModelMissingError):
            parse_descriptor({"framework": "tensorflow"}, tmp_path / "m.onnx")

    def test_missing_weights_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ModelMissingError):
            load_descriptor(tmp_path / "absent.onnx")

    def test_descriptor_is_read_from_beside_the_model(self, tmp_path: Path) -> None:
        weights = tmp_path / "m.onnx"
        weights.write_bytes(b"stub")
        (tmp_path / "m.json").write_text(
            json.dumps({"framework": "onnx", "classes": ["happy", "sad"]}), encoding="utf-8"
        )
        assert load_descriptor(weights).canonical_classes == ("happy", "sad")

    def test_missing_descriptor_defaults_to_primary_classes(self, tmp_path: Path) -> None:
        weights = tmp_path / "m.onnx"
        weights.write_bytes(b"stub")
        descriptor = load_descriptor(weights)
        assert descriptor.canonical_classes == (
            "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"
        )

    def test_corrupt_descriptor_raises(self, tmp_path: Path) -> None:
        weights = tmp_path / "m.onnx"
        weights.write_bytes(b"stub")
        (tmp_path / "m.json").write_text("not json", encoding="utf-8")
        with pytest.raises(ModelMissingError):
            load_descriptor(weights)

    def test_framework_inferred_from_suffix(self, tmp_path: Path) -> None:
        weights = tmp_path / "m.pt"
        weights.write_bytes(b"stub")
        assert load_descriptor(weights).framework == "torchscript"

    def test_preprocess_produces_a_nchw_batch(self, face_crop: np.ndarray) -> None:
        descriptor = ModelDescriptor(
            name="t",
            framework="onnx",
            weights=Path("x"),
            classes=("happy", "sad"),
            input_size=(64, 64),
            scale=1 / 255.0,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        tensor = preprocess(face_crop, descriptor)
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == np.float32
        assert float(tensor.min()) > -2.0 and float(tensor.max()) < 2.0

    def test_preprocess_handles_grayscale(self, gray_frame: np.ndarray) -> None:
        descriptor = ModelDescriptor(
            name="t", framework="onnx", weights=Path("x"), classes=("happy",), input_size=(32, 32)
        )
        assert preprocess(gray_frame, descriptor).shape == (1, 3, 32, 32)


class TestTrainedBackendsWithoutWeights:
    """The trained backends must fail clearly rather than silently degrade."""

    def test_onnx_classifier_without_weights(self) -> None:
        from visionai.ai.models.onnx_classifier import OnnxExpressionClassifier

        with pytest.raises(ModelMissingError):
            OnnxExpressionClassifier(model_path=None).load()

    def test_onnx_classifier_with_missing_file(self, tmp_path: Path) -> None:
        from visionai.ai.models.onnx_classifier import OnnxExpressionClassifier

        with pytest.raises(ModelMissingError):
            OnnxExpressionClassifier(model_path=tmp_path / "gone.onnx").load()

    def test_torch_classifier_without_weights(self) -> None:
        from visionai.ai.models.torch_classifier import TorchScriptExpressionClassifier

        with pytest.raises(ModelMissingError):
            TorchScriptExpressionClassifier(model_path=None).load()

    def test_registry_falls_back_to_heuristic(self, settings, registry: ModelRegistry) -> None:
        settings.models.classifier = "onnx"
        settings.models.classifier_weights = "/nonexistent/model.onnx"
        classifier, report = registry.build_classifier(settings)
        assert classifier.is_heuristic is True
        assert report.degraded is True
        assert "onnx" in report.reason
        classifier.close()

    def test_classifier_ceiling_is_applied_by_the_stage(
        self, settings, registry: ModelRegistry, face_crop: np.ndarray
    ) -> None:
        from visionai.ai.emotion_classifier import ClassificationStage

        classifier, _ = registry.build_classifier(settings)
        stage = ClassificationStage(classifier=classifier, confidence_threshold=0.3)
        result = stage.run(face_crop, [])
        assert result == {}
        assert stage.passes_threshold(
            ExpressionResult(label="happy", confidence=0.9, is_heuristic=True)
        ) is False
        classifier.close()
