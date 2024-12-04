from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from scipy.stats import pearsonr, spearmanr

# Load the STS-B dataset from GLUE
dataset = load_dataset("glue", "stsb")

# Load the pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Regression task
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128
    )

# Apply the preprocessing function to the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset into train and validation sets (validation set is used as test set here)
train_dataset = encoded_dataset["train"]
test_dataset = encoded_dataset["validation"]

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Correct directory path for results
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",  # Prevents logging to external services
    eval_strategy="epoch",  # Evaluate after each epoch
)

# Define evaluation metrics (MSE, Pearson correlation, and Spearman correlation)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze(-1)  # Remove extra dimension for regression output

    # Compute Pearson Correlation, and Spearman Correlation
    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)

    return {
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr
    }

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Replace with your training dataset
    eval_dataset=test_dataset,    # Replace with your evaluation dataset
    data_collator=data_collator,  # Apply dynamic padding
    compute_metrics=compute_metrics  # Define compute_metrics function
)

# Evaluate the model before training (you can also use `trainer.train()` to start training)
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
