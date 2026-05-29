import re

class CleaningAgent:
    def __init__(self, config):
        self.config = config

    def clean(self, text):
        if not isinstance(text, str):
            return text

        # 1. Essential: Normalize whitespace (crucial for BERT tokenization)
        text = " ".join(text.split()).strip()

        # 2. Conditional: Lowercasing (only if using an uncased model, ClinicalBioBert is cased so 'false' toggle)
        if self.config["agent"].get("lowercase", False):
            text = text.lower()

        # 3. Targeted Punctuation: Remove only "noise" symbols, keep sentence markers
        if self.config["agent"].get("remove_noise_chars"):
            # Keeps . , ? ! but removes things like @ # ^ *
            text = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", text)

        return text


