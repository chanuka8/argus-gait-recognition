import shutil
import tempfile

import cv2
import numpy as np
import pytest

from enrollment.appearance_gallery_updater import AppearanceGalleryUpdater
from enrollment.enrollment_manager import EnrollmentManager
from enrollment.gallery_updater import GalleryUpdater
from pipeline.steps.appearance_matching_step import AppearanceMatchingStep
from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep
from storage.vector_store import VectorStore, validate_gallery_files


@pytest.fixture
def temp_gallery_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_appearance_gallery_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_crop():
    # 256x128x3 BGR synthetic image
    return np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)


def test_photo_to_512d_appearance_embedding(sample_crop):
    """Requirement 1 & 2: Extract 512D L2-normalized float32 embedding from photo/crop."""
    extractor = ReIDFeatureExtractionStep()
    embedding = extractor.extract(sample_crop)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32

    # L2 Norm check
    norm = float(np.linalg.norm(embedding))
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_photo_file_extraction(tmp_path, sample_crop):
    """Test extraction directly from an image file path."""
    img_path = tmp_path / "person_photo.jpg"
    cv2.imwrite(str(img_path), sample_crop)

    extractor = ReIDFeatureExtractionStep()
    embedding = extractor.extract(img_path)

    assert embedding is not None
    assert embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_appearance_gallery_single_person_enrollment(temp_gallery_dir, sample_crop):
    """Requirement 3 & 5: Enroll person with single photo and verify persistence."""
    updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)
    extractor = ReIDFeatureExtractionStep()

    emb = extractor.extract(sample_crop)
    updater.add_person("Person_001", [emb])

    # Validate storage persistence
    store = VectorStore(gallery_dir=temp_gallery_dir)
    loaded = store.load()
    assert loaded is not None

    features, labels, metadata = loaded
    assert features.shape == (1, 512)
    assert len(labels) == 1
    assert labels[0] == "Person_001"
    assert "Person_001" in metadata
    assert metadata["Person_001"]["embeddings"] == 1
    assert metadata["Person_001"]["status"] == "ACTIVE"


def test_appearance_gallery_multiple_photos_for_one_person(temp_gallery_dir):
    """Requirement 4: Enrolling multiple reference photos for one person ID."""
    updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)
    extractor = ReIDFeatureExtractionStep()

    crops = [
        np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8),
        np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8),
        np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8),
    ]
    embeddings = [extractor.extract(c) for c in crops]

    updater.add_person("Person_002", embeddings)

    store = VectorStore(gallery_dir=temp_gallery_dir)
    features, labels, metadata = store.load()

    assert features.shape == (3, 512)
    assert list(labels) == ["Person_002", "Person_002", "Person_002"]
    assert metadata["Person_002"]["embeddings"] == 3


def test_appearance_matching_known_match(temp_gallery_dir, sample_crop):
    """Requirement 6 & 7: Cosine similarity matching returns correct known identity."""
    updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)
    extractor = ReIDFeatureExtractionStep()

    target_emb = extractor.extract(sample_crop)
    other_crop = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
    other_emb = extractor.extract(other_crop)

    updater.add_person("Alice", [target_emb])
    updater.add_person("Bob", [other_emb])

    store = VectorStore(gallery_dir=temp_gallery_dir)
    features, labels, metadata = store.load()

    matcher = AppearanceMatchingStep(threshold=0.60)
    matched_id, score = matcher.match(
        query_feature=target_emb,
        gallery_features=features,
        gallery_labels=labels,
        metadata=metadata,
    )

    assert matched_id == "Alice"
    assert np.isclose(score, 1.0, atol=1e-4)


def test_appearance_matching_unknown_below_threshold(temp_gallery_dir):
    """Requirement 8: Explicit UNKNOWN_PERSON return when score is below threshold."""
    updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)

    # Unit vector 1
    v1 = np.zeros((512,), dtype=np.float32)
    v1[0] = 1.0

    # Orthogonal unit vector 2
    v2 = np.zeros((512,), dtype=np.float32)
    v2[1] = 1.0

    updater.add_person("Alice", [v1])

    store = VectorStore(gallery_dir=temp_gallery_dir)
    features, labels, metadata = store.load()

    matcher = AppearanceMatchingStep(threshold=0.60)
    matched_id, score = matcher.match(
        query_feature=v2,
        gallery_features=features,
        gallery_labels=labels,
        metadata=metadata,
        unknown_label="UNKNOWN_PERSON",
    )

    assert matched_id == "UNKNOWN_PERSON"
    assert np.isclose(score, 0.0, atol=1e-5)


def test_appearance_matching_top_k(temp_gallery_dir):
    """Test top-K ranking candidates by similarity."""
    updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)

    v_query = np.zeros((512,), dtype=np.float32)
    v_query[0] = 1.0

    v_high = np.zeros((512,), dtype=np.float32)
    v_high[0] = 0.9
    v_high[1] = np.sqrt(1.0 - 0.9**2)

    v_low = np.zeros((512,), dtype=np.float32)
    v_low[0] = 0.4
    v_low[1] = np.sqrt(1.0 - 0.4**2)

    updater.add_person("HighMatch", [v_high])
    updater.add_person("LowMatch", [v_low])

    store = VectorStore(gallery_dir=temp_gallery_dir)
    features, labels, metadata = store.load()

    matcher = AppearanceMatchingStep()
    top_matches = matcher.top_k_matches(
        query_feature=v_query,
        gallery_features=features,
        gallery_labels=labels,
        metadata=metadata,
        k=2,
    )

    assert len(top_matches) == 2
    assert top_matches[0][0] == "HighMatch"
    assert top_matches[1][0] == "LowMatch"
    assert top_matches[0][1] > top_matches[1][1]


def test_dimension_isolation_rejection(temp_gallery_dir):
    """Requirement 9: Reject 256D vectors in Appearance gallery and 512D vectors in Gait gallery."""
    appearance_updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)
    gait_updater = GalleryUpdater(gallery_dir=temp_gallery_dir)

    vec_256 = np.random.randn(256).astype(np.float32)
    vec_512 = np.random.randn(512).astype(np.float32)

    # Adding 256D vector to AppearanceGalleryUpdater must fail
    with pytest.raises(ValueError, match=r"512-dimensional"):
        appearance_updater.add_person("TestPerson", [vec_256])

    # Adding 512D vector to Gait GalleryUpdater must fail
    with pytest.raises(ValueError, match=r"256-dimensional"):
        gait_updater.add_person("TestPerson", [vec_512])


def test_enrollment_manager_appearance_flow(tmp_path, temp_gallery_dir):
    """Test full EnrollmentManager photo folder appearance enrollment."""
    person_folder = tmp_path / "Subject_42"
    person_folder.mkdir()

    # Create 2 synthetic photos
    img1 = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(person_folder / "p1.jpg"), img1)
    cv2.imwrite(str(person_folder / "p2.png"), img2)

    manager = EnrollmentManager()
    manager.appearance_gallery_updater = AppearanceGalleryUpdater(gallery_dir=temp_gallery_dir)

    res = manager.enroll_person(str(person_folder))
    assert res["success"] is True
    assert res["gallery"] == "appearance"
    assert res["embeddings_added"] == 2

    # Validate vector store files
    valid, _, count = validate_gallery_files(temp_gallery_dir, expected_dim=512)
    assert valid is True
    assert count == 2
