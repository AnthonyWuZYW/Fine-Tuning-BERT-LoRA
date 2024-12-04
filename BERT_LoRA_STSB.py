import time
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
import torch
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
    # In STS-B, handle "sentence1" and "sentence2" fields
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
    output_dir="./results",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
)

# Define evaluation metrics (MSE, Pearson correlation, and Spearman correlation)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Compute Pearson Correlation, and Spearman Correlation
    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)

    return {
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr
    }

# Custom Trainer with proper metric calculation
class MyTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Calling the original evaluate method
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Perform evaluation manually if necessary (e.g., to retrieve logits and labels)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        all_preds, all_labels = [], []
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = {key: value.to(self.args.device) for key, value in batch.items() if key != "labels"}
                outputs = self.model(**inputs)
                logits = outputs.logits
                labels = batch["labels"].to(self.args.device)
                all_preds.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        # Compute metrics
        if self.compute_metrics is not None:
            metrics = self.compute_metrics((all_preds, all_labels))
        
        eval_results.update(metrics)
        
        return eval_results


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