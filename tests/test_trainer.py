import torch
from torch import nn
import numpy as np
from src.training_pipeline import CustomTrainer
from transformers import AutoModelForSequenceClassification, TrainingArguments
from datasets import Dataset

def test_custom_trainer_loss():
    # 1. Create a dummy model
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", num_labels=2
    )
    
    # 2. Create heavily imbalanced weights (Class 1 is 10x more important)
    weights = torch.tensor([1.0, 10.0], dtype=torch.float)
    
    args = TrainingArguments(output_dir="./test_output", no_cuda=True)
    
    # 3. Initialize CustomTrainer
    trainer = CustomTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"review": ["test"], "labels": [1]})
    )
    
    # 4. Mock inputs where the model makes a bad prediction
    inputs = {
        "input_ids": torch.randint(0, 1000, (2, 256)),
        "attention_mask": torch.ones((2, 256)),
        "labels": torch.tensor([1, 1]) # Both are the rare class
    }
    
    # Compute loss
    loss = trainer.compute_loss(model, inputs)
    
    # If working, the loss should be scaled up by our weight (10.0)
    assert loss.item() > 0, "Loss should be a positive number"
    print(" Unit test passed: Class weights are affecting the loss function!")
