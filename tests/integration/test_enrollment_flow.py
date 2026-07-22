from pathlib import Path

import pytest

from enrollment.enrollment_validator import EnrollmentValidator
from storage.vector_store import VectorStore


def test_enrollment_folder_exists(
    enrollment_sample_folder,
):
    folder = Path(enrollment_sample_folder)
    if not folder.exists():
        pytest.skip(f"Enrollment sample folder asset not found: {enrollment_sample_folder}")

    assert folder.exists()


def test_enrollment_validator(
    enrollment_sample_folder,
):
    folder = Path(enrollment_sample_folder)
    if not folder.exists():
        pytest.skip(f"Enrollment sample folder asset not found: {enrollment_sample_folder}")

    validator = EnrollmentValidator()

    valid, message = validator.validate_person_folder(str(enrollment_sample_folder))

    assert valid is True


def test_gallery_load():
    gallery = VectorStore().load()
    if gallery is None:
        pytest.skip("Vector store gallery files not found")

    assert gallery is not None

    features, labels, metadata = gallery

    assert len(features) > 0
    assert len(labels) > 0
    assert len(metadata) > 0
