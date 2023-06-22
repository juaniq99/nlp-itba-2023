import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CUDA test
#x = torch.rand(5, 3)
#print(x)
#print(torch.cuda.is_available())
#exit()

# Using GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Select tokenizer and model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!", "I hate this"]

# Tokenize sentences into model inputs, send inputs and model to GPU
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt").to(device)
model = model.to(device)
print("\nInputs: " + str(tokens) + "\n")

# Get classifications
outputs = model(**tokens)
print("\nOutputs: " + str(outputs) + "\n")

# Get predictions
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
print("\nPredictions: " + str(predictions) + "\n")