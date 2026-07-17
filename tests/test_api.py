import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Mock the model loading BEFORE importing the app
with patch("src.api.download_and_extract_model_from_s3") as mock_download, \
     patch("src.api.AutoTokenizer.from_pretrained") as mock_tokenizer_load, \
     patch("src.api.AutoModelForSequenceClassification.from_pretrained") as mock_model_load:
     
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

# Initialize the automated test client runner
client = TestClient(app)

# =========================================================================
# 1. SYSTEM HEALTH GATES
# =========================================================================
def test_api_liveness_probe():
    """Verify that the system health endpoint returns status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prometheus_metrics_gateway():
    """Verify that Prometheus telemetry metrics are exposed successfully."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


# =========================================================================
# 2. MODEL INFERENCE GATES (MOCKED FOR CI/CD PIPELINES)
# =========================================================================
@patch("src.api.model")
@patch("src.api.tokenizer")
def test_model_inference_positive_adr(mock_tokenizer, mock_model):
    """Verify that a positive adverse drug reaction payload processes cleanly."""
    # Configure mock model to return logits
    mock_output = MagicMock()
    mock_output.logits = MagicMock()
    mock_model.return_value = mock_output
    
    payload = {
        "review": "I took this medication and developed a severe skin rash within an hour."
    }
    response = client.post("/predict", json=payload)

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
    response = client.post("/predict", json=payload)
    
    # If using custom handling, maps to 400. If using Pydantic Field, maps to 422.
    assert response.status_code in [400, 422]


def test_model_inference_excessive_length_payload():
    """Verify that payloads exceeding character constraints are blocked immediately."""
    # Create a malicious payload that exceeds your 1000-character guardrail
    giant_text = "A" * 1500
    payload = {"review": giant_text}
    
    response = client.post("/predict", json=payload)
    
    # Pydantic validation throws a 422 error before your model allocates RAM
    assert response.status_code == 422
    assert "detail" in response.json()


# =========================================================================
# 4. INFRASTRUCTURE & RESOURCE PROTECTION GATES (RATE LIMITING)
# =========================================================================
def test_rate_limiting_activation():
    """Verify that flooding the endpoint triggers an HTTP 429 Too Many Requests error."""
    payload = {"review": "Normal test baseline request sequence."}
    
    # Fire off 10 rapid-fire requests in a loop to intentionally crash the limit
    responses = [client.post("/predict", json=payload) for _ in range(10)]
    
    # Check if any of the later responses register the HTTP 429 safety block
    status_codes = [r.status_code for r in responses]
    
    assert 429 in status_codes
