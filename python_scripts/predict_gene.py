import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Define the topic labels from your training data
# Make sure this list is in the same order as your training data
label_list = [
    'Lichen_Symbiosis_Genomics',
    'Coral_Algae_Symbiosis',
    'Squid_Vibrio_Symbiosis',
    'Plant_Mycorrhizal_Symbiosis',
    'Legume_Rhizobium_Symbiosis',
    'Human_Microbiome',
    'Environmental_Microbes_Ecology',
    'General_Gene_Expression',
    'Sponge_Symbiosis_Genomics',
    'Common_Bean_Nitrogen_Fixation'
]

# Load the model and tokenizer from your saved directory
model_dir = "/scratch/pperez40/symbiosis_llm/results/checkpoint-173"  # Use the last checkpoint folder from your output
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a new abstract to classify
new_abstract = "The symbiotic relationship between the bobtail squid and the bacterium Vibrio fischeri is essential for the host to produce bioluminescence."

# Tokenize the new abstract
inputs = tokenizer(new_abstract, return_tensors="pt", truncation=True, padding=True)
inputs.to(device)

# Get model predictions
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted class ID
predicted_class_id = torch.argmax(logits, dim=1).item()
predicted_label = label_list[predicted_class_id]

# Print the result
print(f"\n--- Prediction Result ---")
print(f"Abstract: {new_abstract}")
print(f"Predicted Topic: {predicted_label}")
print(f"-------------------------\n")
