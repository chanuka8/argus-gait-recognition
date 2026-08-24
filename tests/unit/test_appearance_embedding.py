import numpy as np

from intelligence.appearance_embedding import AppearanceEmbeddingExtractor


def test_appearance_embedding_initialization():
    extractor = AppearanceEmbeddingExtractor(update_interval=5)
    assert extractor.update_interval == 5


def test_appearance_embedding_normalization_and_gating():
    extractor = AppearanceEmbeddingExtractor(update_interval=5)

    crop = np.full((100, 50, 3), 128, dtype=np.uint8)

    emb1 = extractor.extract(crop, track_id=1, frame_index=0)
    if extractor.is_available() and emb1 is not None:
        assert emb1.shape == (512,)
        assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-5)

    emb2 = extractor.extract(crop, track_id=1, frame_index=2)
    if emb1 is not None:
        assert np.allclose(emb1, emb2)

    emb_unreliable = extractor.extract(crop, track_id=1, frame_index=10, track_reliable=False)
    if emb1 is not None:
        assert np.allclose(emb1, emb_unreliable)

    extractor.clear_track(1)
    assert extractor.get_cached(1) is None
