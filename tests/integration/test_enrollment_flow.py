from pathlib import Path

from enrollment.enrollment_validator import EnrollmentValidator
from storage.vector_store import VectorStore


def test_enrollment_folder_exists(
    enrollment_sample_folder,
):
    assert Path(
        enrollment_sample_folder
    ).exists()


def test_enrollment_validator(
    enrollment_sample_folder,
):
    validator = EnrollmentValidator()

    valid, message = (
        validator.validate_person_folder(
            str(enrollment_sample_folder)
        )
    )

    assert valid is True


def test_gallery_load():
    gallery = VectorStore().load()

    assert gallery is not None

    features, labels, metadata = gallery

    assert len(features) > 0
    assert len(labels) > 0
    assert len(metadata) > 0