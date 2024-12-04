from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)
from sklearn.metrics import matthews_corrcoef, accuracy_score
from Trainer import MyTrainer

# Load the CoLA dataset from GLUE
dataset = load_dataset("glue", "cola")


# Load the pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    output_dir="./results",  # Correct directory path
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
)

# Define evaluation metrics including MCC and accuracy
def compute_metrics(eval_pred):
    print("Computing metrics...") 
    predictions, labels = eval_pred
    # Compute MCC and accuracy
    matthews_corr = matthews_corrcoef(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    return {
        "matthews_correlation": matthews_corr,
        "accuracy": accuracy
    }


# Set up trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Replace with your training dataset
    eval_dataset=test_dataset,    # Replace with your evaluation dataset
    compute_metrics=compute_metrics  # Define compute_metrics function
)

# Evaluate the model before training
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
