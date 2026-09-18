from fastapi.testclient import TestClient
from api.app import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.3.0"


def test_root_redirects_to_local_app():
    client = TestClient(app, follow_redirects=False)
    response = client.get("/")
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/app/"
