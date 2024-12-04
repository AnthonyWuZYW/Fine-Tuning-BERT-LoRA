import time
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
from Trainer import MyTrainer

# Load the STS-B dataset from GLUE
dataset = load_dataset("glue", "stsb")

import torch
import time
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Load the STS-B dataset from GLUE
dataset = load_dataset("glue", "stsb")

# Load the pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Single regression output
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout
    target_modules=["query", "key", "value"]  # Apply LoRA to specific Linear layers
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

def preprocess_function(examples):
    # In STS-B, we need to handle "sentence1" and "sentence2" fields
    return tokenizer(
        examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128
    )

# Preprocess and encode the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset into train and test sets
train_dataset = encoded_dataset["train"]
test_dataset = encoded_dataset["validation"]

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Correct directory path
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
)

# Define evaluation metrics for regression
def compute_metrics(eval_pred):
    print("Computing metrics...")
    predictions, labels = eval_pred
    predictions = predictions.flatten()  # Ensure shape compatibility
    pearson_corr = pearsonr(predictions, labels)[0]
    spearman_corr = spearmanr(predictions, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }


# Set up trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Measure time for training
start_time = time.time()

# Evaluate the model before training
eval_results = trainer.evaluate()
print("Evaluation results before training:", eval_results)

# Train the model
trainer.train()

# Evaluate the model after training
eval_results = trainer.evaluate()

# Measure total time
end_time = time.time()
total_time = end_time - start_time

print("Evaluation results after training:", eval_results)
print(f"Total time taken for training and evaluation: {total_time:.2f} seconds")

# Save the LoRA-adapted model
model.save_pretrained("./lora_bert_stsb")
tokenizer.save_pretrained("./lora_bert_stsb")