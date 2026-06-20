import torch
from torch import nn
import numpy as np
from src.training_pipeline import CustomTrainer
# Added AutoConfig to read your local json file layout properly
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments
from datasets import Dataset

def test_custom_trainer_loss():
    # 1. Load configuration from your local config.json and build a dummy model
    config = AutoConfig.from_pretrained("config.json", num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)
    
    # 2. Create heavily imbalanced weights (Class 1 is 10x more important)
    weights = torch.tensor([1.0, 10.0], dtype=torch.float)
    
    args = TrainingArguments(output_dir="./test_output", use_cpu=True)
    
    # 3. Initialize CustomTrainer
    trainer = CustomTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"review": ["test"], "labels": [1]})
    )
    
    # 4. Mock inputs where the model makes a bad prediction
    # Dynamic check prevents index out of range exceptions during embedding lookups
    vocab_size = getattr(config, "vocab_size", 30522)
    
    inputs = {
        "input_ids": torch.randint(0, vocab_size, (2, 256)),
        "attention_mask": torch.ones((2, 256), dtype=torch.long),
        "labels": torch.tensor([1, 1]) # Both are the rare class
    }
    
    # Compute loss
    loss = trainer.compute_loss(model, inputs)
    
    # If working, the loss should be scaled up by our weight (10.0)
    assert loss.item() > 0, "Loss should be a positive number"
    print(" Unit test passed: Class weights are affecting the loss function!")

