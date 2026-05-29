import pytest
# Explicitly import from the clean src package layout
from src.cleaning_agent import CleaningAgent


@pytest.fixture
def base_config():
    """Provides a default baseline pipeline configuration dictionary."""
    return {
        "agent": {
            "lowercase": False,
            "remove_noise_chars": True
        }
    }


def test_whitespace_normalization(base_config):
    """Verifies that trailing, leading, and duplicated spaces are compacted."""
    agent = CleaningAgent(base_config)
    dirty_text = "   Patient   presented   with severe   headaches.   "
    expected = "Patient presented with severe headaches."
    
    assert agent.clean(dirty_text) == expected


def test_lowercase_toggle_false(base_config):
    """Confirms casing is fully preserved when lowercase config is False."""
    base_config["agent"]["lowercase"] = False
    agent = CleaningAgent(base_config)
    text = "The Patient was prescribed Metformin."
    
    assert agent.clean(text) == "The Patient was prescribed Metformin."


def test_lowercase_toggle_true(base_config):
    """Verifies text strings are completely lowercased when toggle is True."""
    base_config["agent"]["lowercase"] = True
    agent = CleaningAgent(base_config)
    text = "The Patient was prescribed Metformin."
    
    assert agent.clean(text) == "the patient was prescribed metformin."


def test_noise_character_removal(base_config):
    """Ensures junk symbols are stripped but vital sentence markers are kept intact."""
    base_config["agent"]["remove_noise_chars"] = True
    agent = CleaningAgent(base_config)
    dirty_text = "Alert! Patient's blood pressure was over 140/90 #critical @hospital**"
    expected = "Alert! Patients blood pressure was over 14090 critical hospital"
    
    assert agent.clean(dirty_text) == expected


def test_noise_character_toggle_false(base_config):
    """Confirms noise symbols are left untouched if the feature is disabled."""
    base_config["agent"]["remove_noise_chars"] = False
    agent = CleaningAgent(base_config)
    text = "Severe pain scale 10/10 #fibromyalgia"
    
    assert agent.clean(text) == "Severe pain scale 10/10 #fibromyalgia"


@pytest.mark.parametrize("non_string_input", [None, 12345, 4.5, [], {}])
def test_non_string_input_handling(base_config, non_string_input):
    """Guarantees the agent gracefully returns the input if a non-string object is passed."""
    agent = CleaningAgent(base_config)
    
    assert agent.clean(non_string_input) == non_string_input
