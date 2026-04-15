import random

def introduce_noise(text):

    if not isinstance(text, str):
        return text

    if random.random() < 0.3:
        text = text.lower()

    if random.random() < 0.3:
        text = text.replace("e", "")

    return text