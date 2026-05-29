import torch
from transformers import pipeline

def load_ner_pipeline():
    """Initializes the Biomedical NER pipeline based on available local hardware."""
    device = 0 if torch.cuda.is_available() else -1  # Dynamic fallback to CPU (-1)
    print(f"Initializing Biomedical NER pipeline on device: {'GPU (0)' if device == 0 else 'CPU (-1)'}")
    
    return pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=device
    )

def enrich_and_label_batched(examples, ner_pipeline):
    """
    Enriches incoming data matrices by appending clinical entities 
    and assigning a binary ADR target classification.
    """
    # Dynamic batch resizing to protect limited CPU memory
    is_gpu = next(ner_pipeline.model.parameters()).is_cuda
    inference_batch_size = 64 if is_gpu else 2  # Drop to batch size 2 on CPU

    batch_entities = ner_pipeline(
        examples["review"],
        batch_size=inference_batch_size
    )

    new_reviews = []
    adr_labels = []

    for text, entities in zip(examples["review"], batch_entities):
        entity_text = " ".join(ent["word"] for ent in entities)
        has_adr = any(ent["entity_group"] == "Sign_symptom" for ent in entities)

        new_reviews.append(f"{text} [ENT] {entity_text}")
        adr_labels.append(int(has_adr))

    examples["review"] = new_reviews
    examples["adr_label"] = adr_labels
    return examples

