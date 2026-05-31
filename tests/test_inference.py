import pytest
import requests

# Define the target server address constants explicitly
BASE_URL = "http://127.0.0.1:8000"


def test_api_liveness_probe():
    """Verify that the system health endpoint returns status healthy."""
    target_url = f"{BASE_URL}/health"
    response = requests.get(target_url)

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prometheus_metrics_gateway():
    """Verify that Prometheus telemetry metrics are exposed successfully."""
    target_url = f"{BASE_URL}/metrics"
    response = requests.get(target_url)

    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_model_inference_positive_adr():
    """Verify that a positive adverse drug reaction payload processes cleanly."""
    target_url = f"{BASE_URL}/predict"
    payload = {
        "review": "I took this medication and developed a severe skin rash within an hour."
    }

    response = requests.post(target_url, json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction_class_id" in data
    assert "class_probabilities" in data


def test_model_inference_empty_payload():
    """Verify that empty request inputs are caught by system validation handling."""
    target_url = f"{BASE_URL}/predict"
    payload = {"review": "   "}

    response = requests.post(target_url, json=payload)
    assert response.status_code == 400



