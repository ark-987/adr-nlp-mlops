class CleaningAgent:

    def __init__(self, config):
        self.config = config

    def clean(self, text):

        if not isinstance(text, str):
            return text

        text = text.lower().strip()

        if self.config["agent"].get("remove_punctuation"):
            import re
            text = re.sub(r"[^\w\s]", "", text)
        return text