# classifier.py
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import os

# Define dataset class
class IssueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class IssueClassifier:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.category_model_path = 'models/category_model'
        self.urgency_model_path = 'models/urgency_model'
        
        # Category mapping
        self.categories = {
            0: 'plumbing',
            1: 'electrical',
            2: 'hvac',
            3: 'structural',
            4: 'appliance',
            5: 'pest'
        }
        self.category_mapping = {v: k for k, v in self.categories.items()}
        
        # Urgency mapping
        self.urgencies = {
            0: 'non-urgent',
            1: 'urgent'
        }
        self.urgency_mapping = {v: k for k, v in self.urgencies.items()}
        
        # Initialize models
        self.category_model = None
        self.urgency_model = None
        
        # Load models if they exist
        self.load_models()
    
    def load_models(self):
        # Load category model if it exists
        if os.path.exists(self.category_model_path):
            self.category_model = DistilBertForSequenceClassification.from_pretrained(
                self.category_model_path, 
                num_labels=len(self.categories)
            )
        
        # Load urgency model if it exists
        if os.path.exists(self.urgency_model_path):
            self.urgency_model = DistilBertForSequenceClassification.from_pretrained(
                self.urgency_model_path, 
                num_labels=len(self.urgencies)
            )
    
    def train(self, data_file):
        # Load and preprocess the data
        df = pd.read_csv(data_file)
        
        # Prepare category data
        category_texts = df['text'].tolist()
        category_labels = [self.category_mapping[category] for category in df['category']]
        
        # Prepare urgency data
        urgency_texts = df['text'].tolist()
        urgency_labels = [self.urgency_mapping[urgency] for urgency in df['urgency']]
        
        # Train category model
        print("Training category classifier...")
        self.category_model = self._train_model(
            category_texts, 
            category_labels, 
            len(self.categories), 
            self.category_model_path
        )
        
        # Train urgency model
        print("Training urgency classifier...")
        self.urgency_model = self._train_model(
            urgency_texts, 
            urgency_labels, 
            len(self.urgencies), 
            self.urgency_model_path
        )
        
        print("Training complete")
    
    def _train_model(self, texts, labels, num_labels, output_dir):
        # Initialize model
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=num_labels
        )
        
        # Create dataset
        train_dataset = IssueDataset(texts, labels, self.tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model.save_pretrained(output_dir)
        
        return model
    
    def predict(self, text):
        # Ensure models are loaded
        if self.category_model is None or self.urgency_model is None:
            return "uncategorized", "unknown"
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Predict category
        with torch.no_grad():
            category_outputs = self.category_model(**inputs)
            category_predictions = torch.nn.functional.softmax(category_outputs.logits, dim=-1)
            category_idx = category_predictions.argmax().item()
            category = self.categories[category_idx]
        
        # Predict urgency
        with torch.no_grad():
            urgency_outputs = self.urgency_model(**inputs)
            urgency_predictions = torch.nn.functional.softmax(urgency_outputs.logits, dim=-1)
            urgency_idx = urgency_predictions.argmax().item()
            urgency = self.urgencies[urgency_idx]
        
        return category, urgency

# If a photo is present, we consider it more urgent (rule-based for MVP)
def determine_urgency(text, image_path, classifier):
    # Get predicted urgency from text
    _, predicted_urgency = classifier.predict(text)
    
    # If there's an image and the model didn't already classify as urgent, 
    # we'll increase urgency a bit (simple rule for MVP)
    if image_path and predicted_urgency != "urgent":
        # Check for urgent keywords (simple rule-based approach)
        urgent_keywords = ["leak", "flood", "broken", "emergency", "fire", "smoke", "gas"]
        if any(keyword in text.lower() for keyword in urgent_keywords):
            return "urgent"
    
    return predicted_urgency

# Main function to classify the issue
def classify_issue(text):
    classifier = IssueClassifier()
    category, _ = classifier.predict(text)
    return category

# Usage example:
if __name__ == "__main__":
    # Train models if mock data exists
    classifier = IssueClassifier()
    if os.path.exists('mock_issues.csv'):
        classifier.train('mock_issues.csv')
    
    # Test prediction
    test_text = "The sink in my bathroom is leaking water all over the floor"
    category, urgency = classifier.predict(test_text)
    print(f"Text: {test_text}")
    print(f"Predicted Category: {category}")
    print(f"Predicted Urgency: {urgency}")
    
    # Test with image
    final_urgency = determine_urgency(test_text, "sample_image.jpg", classifier)
    print(f"Final Urgency (with image): {final_urgency}")