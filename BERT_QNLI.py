from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import matthews_corrcoef, accuracy_score

# Load the QNLI dataset from GLUE
dataset = load_dataset("glue", "qnli")

# Load the pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(
        examples["question"], examples["sentence"], padding="max_length", truncation=True, max_length=128
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

# Define evaluation metrics (MCC and accuracy)
def compute_metrics(eval_pred):
    print("Computing metrics...") 
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)  # Convert logits to predicted class labels (0 or 1)
    
    # Ensure labels are in integer format (they may be returned as float32)
    labels = labels.astype(int)
    
    # Compute Accuracy
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy
    }

# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=test_dataset,    
    data_collator=data_collator,  
    compute_metrics=compute_metrics  
)

# Evaluate the model before training 
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)