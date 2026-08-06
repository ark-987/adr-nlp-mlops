import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch

# Mock the model loading BEFORE importing the app
with patch("src.api.download_and_extract_model_from_s3") as mock_download, \
     patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load, \
     patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model_load:
    
    # Configure mocks to return successfully
    mock_download.return_value = True
    mock_tokenizer_load.return_value = MagicMock()
    mock_model_load.return_value = MagicMock()
    
    # Now import the app after mocks are in place
    from src.api import app
    
    # IMPORTANT: Set the global variables after import to ensure they're available
    import src.api
    src.api.model = MagicMock()
    src.api.tokenizer = MagicMock()


# =========================================================================
# 1. SYSTEM HEALTH GATES
# =========================================================================
def test_api_liveness_probe():
    """Verify that the system health endpoint returns status healthy."""
    # Create TestClient inside the test so startup runs after any patching
    with TestClient(app) as client:
        response = client.get("/health")
        if response.status_code != 200:
            print("LIVENESS STATUS:", response.status_code)
            try:
                print("LIVENESS JSON:", response.json())
            except Exception:
                print("LIVENESS TEXT:", response.text)
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_prometheus_metrics_gateway():
    """Verify that Prometheus telemetry metrics are exposed successfully."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        if response.status_code != 200:
            print("METRICS STATUS:", response.status_code)
            print("METRICS TEXT:", response.text)
        assert response.status_code == 200
        assert "http_requests_total" in response.text


# =========================================================================
# 2. MODEL INFERENCE GATES (MOCKED FOR CI/CD PIPELINES)
# =========================================================================
@patch("torch.softmax")
@patch("torch.argmax")
def test_model_inference_positive_adr(mock_argmax, mock_softmax):
    """Verify that a positive adverse drug reaction payload processes cleanly."""
    # Configure mocks so app startup or handlers can use them
    mock_model.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()

    payload = {
        "review": "I took this medication and developed a severe skin rash within an hour."
    }
    # Ensure the TestClient is created after the patches are active
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        if response.status_code != 200:
            print("PREDICT STATUS:", response.status_code)
            try:
                print("PREDICT JSON:", response.json())
            except Exception:
                print("PREDICT TEXT:", response.text)
        assert response.status_code == 200
        data = response.json()
        assert "prediction_class_id" in data
        assert "class_probabilities" in data


# =========================================================================
# 3. DATA QUALITY & INPUT SECURITY GATES
# =========================================================================
def test_model_inference_empty_payload():
    """Verify that empty request inputs are caught by system validation handling."""
    payload = {"review": "   "}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        if response.status_code not in [400, 422]:
            print("EMPTY PAYLOAD STATUS:", response.status_code)
            try:
                print("EMPTY PAYLOAD JSON:", response.json())
            except Exception:
                print("EMPTY PAYLOAD TEXT:", response.text)
        # If using custom handling, maps to 400. If using Pydantic Field, maps to 422.
        assert response.status_code in [400, 422]


def test_model_inference_excessive_length_payload():
    """Verify that payloads exceeding character constraints are blocked immediately."""
    # Create a malicious payload that exceeds your 1000-character guardrail
    giant_text = "A" * 1500
    payload = {"review": giant_text}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        if response.status_code != 422:
            print("EXCESS LENGTH STATUS:", response.status_code)
            try:
                print("EXCESS LENGTH JSON:", response.json())
            except Exception:
                print("EXCESS LENGTH TEXT:", response.text)
        # Pydantic validation throws a 422 error before your model allocates RAM
        assert response.status_code == 422
        assert "detail" in response.json()


# =========================================================================
# 4. INFRASTRUCTURE & RESOURCE PROTECTION GATES (RATE LIMITING)
# =========================================================================
def test_rate_limiting_activation():
    """Verify that flooding the endpoint triggers an HTTP 429 Too Many Requests error."""
    payload = {"review": "Normal test baseline request sequence."}

    with TestClient(app) as client:
        # Fire off 10 rapid-fire requests in a loop to intentionally crash the limit
        responses = [client.post("/predict", json=payload) for _ in range(10)]

        # Check if any of the later responses register the HTTP 429 safety block
        status_codes = [r.status_code for r in responses]

        assert 429 in status_codes
