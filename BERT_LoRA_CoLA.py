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
from sklearn.metrics import matthews_corrcoef, accuracy_score

# Load the CoLA dataset from GLUE
dataset = load_dataset("glue", "cola")


# Load the pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
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
    return tokenizer(
        examples["sentence"], padding="max_length", truncation=True, max_length=128
    )

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
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
)

# Define evaluation metrics including MCC and accuracy
def compute_metrics(eval_pred):
    print("Computing metrics...") 
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    # Compute MCC and accuracy
    matthews_corr = matthews_corrcoef(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    return {
        "matthews_correlation": matthews_corr,
        "accuracy": accuracy
    }

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Replace with your training dataset
    eval_dataset=test_dataset,    # Replace with your evaluation dataset
    compute_metrics=compute_metrics  # Define compute_metrics function
)

# Measure time for training
start_time = time.time()

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Measure total time
end_time = time.time()
total_time = end_time - start_time

print("Evaluation results:", eval_results)
print(f"Total time taken: {total_time:.2f} seconds")

# Save the LoRA-adapted model
model.save_pretrained("./lora_bert_cola")
tokenizer.save_pretrained("./lora_bert_cola")

