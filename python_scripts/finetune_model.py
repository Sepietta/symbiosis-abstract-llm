import pandas as pd
from datasets import Dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate

# Load the labeled data
df = pd.read_csv('/scratch/pperez40/symbiosis_llm/data_pre_processing/labeled_symbiosis_data.csv')

# Drop any rows with missing cleaned text
df.dropna(subset=['cleaned_text'], inplace=True)

# Count the number of articles per topic
topic_counts = df['topic_label'].value_counts()

# Identify topics with only one article
topics_to_remove = topic_counts[topic_counts <= 1].index

# Filter the DataFrame to keep only topics with more than one article
df = df[~df['topic_label'].isin(topics_to_remove)].copy()

# Map topic labels to integers
label_list = df['topic_label'].unique().tolist()
label_map = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_map)
df['label_id'] = df['topic_label'].map(label_map)

# Split the data into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)

# Correct the index issue on the split DataFrames
train_df.reset_index(drop=True, inplace=True)
eval_df.reset_index(drop=True, inplace=True)

# Rename the label_id column to labels for the Trainer
train_df.rename(columns={'label_id': 'labels'}, inplace=True)
eval_df.rename(columns={'label_id': 'labels'}, inplace=True)

# Convert pandas DataFrames to Hugging Face Dataset objects
features = Features({
    'pmid': Value('int64'),
    'cleaned_text': Value('string'),
    'topic_label': Value('string'),
    'labels': ClassLabel(names=label_list)
})

train_dataset = Dataset.from_pandas(train_df, features=features)
eval_dataset = Dataset.from_pandas(eval_df, features=features)

# Load the tokenizer and model
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Pre-process the datasets (tokenize the text)
def tokenize_function(examples):
    return tokenizer(examples['cleaned_text'], truncation=True, padding='max_length')

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/pperez40/symbiosis_llm/results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# Define a function to compute evaluation metrics
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# The model and its files will automatically be saved to the output_dir
print("\nFine-tuning complete. Model and training outputs saved to '/scratch/pperez40/symbiosis_llm/results'")
