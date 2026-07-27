"""Unit tests for Dual-Modal Fusion intelligence components."""

import numpy as np
import pytest

from intelligence.dual_modal_fusion import DualModalFusion
from intelligence.fusion_weights import DynamicFusionWeights
from intelligence.quality_assessment import QualityAssessment
from intelligence.score_normalizer import ScoreNormalizer


class TestScoreNormalizer:
    def test_gait_score_normalization(self):
        normalizer = ScoreNormalizer()
        assert normalizer.normalize_gait(0.8) == pytest.approx(0.8)
        assert normalizer.normalize_gait(1.5) == 1.0
        assert normalizer.normalize_gait(-0.5) == 0.0
        assert normalizer.normalize_gait(None) is None

    def test_reid_score_normalization(self):
        normalizer = ScoreNormalizer(reid_min_max=(-1.0, 1.0))
        assert normalizer.normalize_reid(0.0) == pytest.approx(0.5)
        assert normalizer.normalize_reid(1.0) == pytest.approx(1.0)
        assert normalizer.normalize_reid(-1.0) == pytest.approx(0.0)
        assert normalizer.normalize_reid(None) is None


class TestDynamicFusionWeights:
    def test_weight_constraint_sum_to_one(self):
        weights = DynamicFusionWeights(default_gait_weight=0.7, default_reid_weight=0.3)
        w_gait, w_reid = weights.compute_weights(gait_available=True, reid_available=True)
        assert w_gait + w_reid == pytest.approx(1.0)
        assert w_gait == pytest.approx(0.7)
        assert w_reid == pytest.approx(0.3)

    def test_single_modality_fallback(self):
        weights = DynamicFusionWeights(default_gait_weight=0.7, default_reid_weight=0.3)

        # Gait only available
        w_gait, w_reid = weights.compute_weights(gait_available=True, reid_available=False)
        assert w_gait == 1.0
        assert w_reid == 0.0

        # ReID only available
        w_gait, w_reid = weights.compute_weights(gait_available=False, reid_available=True)
        assert w_gait == 0.0
        assert w_reid == 1.0

    def test_quality_adaptive_weighting(self):
        weights = DynamicFusionWeights(default_gait_weight=0.5, default_reid_weight=0.5)
        # Gait quality is high (1.0), ReID quality is low (0.2)
        w_gait, w_reid = weights.compute_weights(
            gait_available=True,
            reid_available=True,
            gait_quality=1.0,
            reid_quality=0.2,
        )
        assert w_gait > w_reid
        assert w_gait + w_reid == pytest.approx(1.0)


class TestQualityAssessment:
    def test_reid_crop_quality(self):
        qa = QualityAssessment()
        # Invalid crops
        assert qa.evaluate_reid_quality(None) == 0.0
        assert qa.evaluate_reid_quality(np.zeros((10, 10, 3), dtype=np.uint8)) == 0.0

        # Valid crop
        valid_crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        score = qa.evaluate_reid_quality(valid_crop, confidence=0.9)
        assert 0.0 < score <= 1.0

    def test_gait_quality(self):
        qa = QualityAssessment()
        assert qa.evaluate_gait_quality(gei_frame_count=0) == 0.0

        # Full GEI sequence
        dummy_gei = np.zeros((128, 128), dtype=np.uint8)
        dummy_gei[20:80, 40:80] = 255
        score = qa.evaluate_gait_quality(gei_frame_count=30, gei=dummy_gei, confidence=0.95)
        assert 0.0 < score <= 1.0


class TestDualModalFusion:
    def test_fusion_both_modalities(self):
        fusion = DualModalFusion(default_gait_weight=0.6, default_reid_weight=0.4)
        crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        gei = np.zeros((128, 128), dtype=np.uint8)
        gei[20:80, 40:80] = 255

        result = fusion.fuse(
            gait_score=0.85,
            reid_score=0.60,
            crop=crop,
            gei_frame_count=30,
            gei=gei,
        )

        assert "final_score" in result
        assert 0.0 <= result["final_score"] <= 1.0
        assert result["gait_weight"] + result["reid_weight"] == pytest.approx(1.0)
        assert result["active_modalities"] == ["gait", "reid"]

    def test_fusion_single_modality_fallback(self):
        fusion = DualModalFusion()
        result = fusion.fuse(
            gait_score=0.90,
            reid_score=None,
        )
        assert result["final_score"] == pytest.approx(0.90)
        assert result["gait_weight"] == 1.0
        assert result["reid_weight"] == 0.0
        assert result["active_modalities"] == ["gait"]

    def test_cosine_similarity_computation(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        assert DualModalFusion.compute_cosine_similarity(v1, v2) == pytest.approx(1.0)
        assert DualModalFusion.compute_cosine_similarity(v1, v3) == pytest.approx(0.0)
        assert DualModalFusion.compute_cosine_similarity(v1, None) is None

    def test_fusion_with_embeddings(self):
        fusion = DualModalFusion(default_gait_weight=0.5, default_reid_weight=0.5)
        g1 = np.array([1.0, 0.0], dtype=np.float32)
        g2 = np.array([1.0, 0.0], dtype=np.float32)
        r1 = np.array([0.0, 1.0], dtype=np.float32)
        r2 = np.array([0.0, 1.0], dtype=np.float32)

        res = fusion.fuse(
            gait_embedding=g1,
            gait_gallery_embedding=g2,
            reid_embedding=r1,
            reid_gallery_embedding=r2,
        )
        assert res["gait_score_norm"] == pytest.approx(1.0)
        assert res["reid_score_norm"] == pytest.approx(1.0)
        assert res["final_score"] == pytest.approx(1.0)

    def test_from_config_and_is_enabled(self):
        cfg = {"enabled": True, "gait_weight": 0.8, "reid_weight": 0.2}
        fusion = DualModalFusion.from_config(cfg)
        assert fusion.is_enabled() is True
        assert fusion.weight_allocator.base_gait_weight == pytest.approx(0.8)
        assert fusion.weight_allocator.base_reid_weight == pytest.approx(0.2)
