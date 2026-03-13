from fastapi.testclient import TestClient
from main import app

# Create a simulated client to test the API without starting a real server
client = TestClient(app)

def test_api_health():
    # Test if the Swagger UI documentation page loads successfully
    response = client.get("/docs")
    assert response.status_code == 200, "API failed to start or routing is broken."

def test_invalid_endpoint():
    # Test that the API correctly handles bad requests
    response = client.get("/this-does-not-exist")
    assert response.status_code == 404