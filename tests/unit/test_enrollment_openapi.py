import io
from fastapi.testclient import TestClient

from api.server import app
from api.v1.router import get_gait_service


class StubGaitService:
    """Stub service for unit testing the enrollment endpoint contract without invoking ML models."""

    def __init__(self):
        self.recorded_person_id = None
        self.recorded_image_bytes = []

    def enroll_images(self, person_id: str, image_bytes_list: list[bytes]) -> dict:
        self.recorded_person_id = person_id
        self.recorded_image_bytes = image_bytes_list
        return {
            "success": True,
            "person_id": person_id,
            "message": f"Successfully enrolled {person_id} with {len(image_bytes_list)} embeddings",
            "embeddings_added": len(image_bytes_list),
        }


def test_enrollment_openapi_schema():
    """Verify OpenAPI schema reports files as array of binary strings for Swagger file picker."""
    app.openapi_schema = None
    openapi = app.openapi()

    assert "/api/v1/enroll" in openapi.get("paths", {})
    enroll_post = openapi["paths"]["/api/v1/enroll"].get("post", {})
    assert enroll_post is not None

    req_body = enroll_post.get("requestBody", {})
    content = req_body.get("content", {})
    assert "multipart/form-data" in content

    schema_ref = content["multipart/form-data"].get("schema", {}).get("$ref", "")
    assert schema_ref.startswith("#/components/schemas/")
    ref_name = schema_ref.split("/")[-1]

    schemas = openapi.get("components", {}).get("schemas", {})
    assert ref_name in schemas
    enroll_body_schema = schemas[ref_name]

    assert enroll_body_schema.get("type") == "object"
    assert "properties" in enroll_body_schema
    assert "files" in enroll_body_schema["properties"]

    files_schema = enroll_body_schema["properties"]["files"]
    assert files_schema.get("type") == "array"
    assert "items" in files_schema
    assert files_schema["items"].get("type") == "string"
    assert files_schema["items"].get("format") == "binary"


def test_enrollment_multipart_endpoint_contract():
    """Verify multipart/form-data upload receives person_id and multiple image byte streams."""
    stub_service = StubGaitService()

    app.dependency_overrides[get_gait_service] = lambda: stub_service

    try:
        client = TestClient(app)

        file1_content = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00test_image_1"
        file2_content = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00test_image_2"

        files = [
            ("files", ("image1.jpg", io.BytesIO(file1_content), "image/jpeg")),
            ("files", ("image2.jpg", io.BytesIO(file2_content), "image/jpeg")),
        ]
        data = {
            "person_id": "test_person_42",
        }

        response = client.post("/api/v1/enroll", data=data, files=files)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        res_json = response.json()

        assert res_json.get("success") is True
        assert res_json.get("person_id") == "test_person_42"
        assert res_json.get("embeddings_added") == 2

        assert stub_service.recorded_person_id == "test_person_42"
        assert len(stub_service.recorded_image_bytes) == 2
        assert stub_service.recorded_image_bytes[0] == file1_content
        assert stub_service.recorded_image_bytes[1] == file2_content
    finally:
        app.dependency_overrides.pop(get_gait_service, None)
