#!/usr/bin/env python3
"""
BERT Fine-Tuning Pipeline for Fake/Real Text Classification
This pipeline implements BERT fine-tuning on the actual text data to improve model performance.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """Custom dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTFineTuningPipeline:
    """Main pipeline for BERT fine-tuning"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
        print(f"Using device: {self.device}")
        print(f"BERT model: {model_name}")
    
    def load_text_data(self):
        """Load and preprocess the actual text data"""
        print("Loading text data...")
        
        # Load training labels
        train_labels = pd.read_csv('data/train.csv')
        print(f"Training labels shape: {train_labels.shape}")
        
        # Load training texts
        train_texts = []
        train_labels_list = []
        
        for idx, row in train_labels.iterrows():
            article_id = row['id']
            real_text_id = row['real_text_id']
            
            # Convert to article directory name
            article_dir = f"article_{article_id:04d}"
            article_path = os.path.join('data/train', article_dir)
            
            if os.path.exists(article_path):
                # Read all text files in the article directory
                article_texts = []
                for file_name in os.listdir(article_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(article_path, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                article_texts.append(text)
                
                if article_texts:
                    # Combine all texts for this article
                    combined_text = ' [SEP] '.join(article_texts)
                    train_texts.append(combined_text)
                    # Convert real_text_id to binary: 1 = real, 2 = fake
                    label = 1 if real_text_id == 1 else 0
                    train_labels_list.append(label)
        
        print(f"Loaded {len(train_texts)} training samples")
        print(f"Label distribution: {np.bincount(train_labels_list)}")
        
        return train_texts, train_labels_list
    
    def load_test_data(self):
        """Load test data for predictions"""
        print("Loading test data...")
        
        test_texts = []
        test_article_ids = []
        
        test_dir = 'data/test'
        for article_dir in sorted(os.listdir(test_dir)):
            if article_dir.startswith('article_'):
                article_path = os.path.join(test_dir, article_dir)
                article_id = int(article_dir.replace('article_', ''))
                
                # Read all text files in the article directory
                article_texts = []
                for file_name in os.listdir(article_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(article_path, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                article_texts.append(text)
                
                if article_texts:
                    # Combine all texts for this article
                    combined_text = ' [SEP] '.join(article_texts)
                    test_texts.append(combined_text)
                    test_article_ids.append(article_id)
        
        print(f"Loaded {len(test_texts)} test samples")
        return test_texts, test_article_ids
    
    def initialize_model(self):
        """Initialize BERT model and tokenizer"""
        print(f"Initializing {self.model_name}...")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        self.model.to(self.device)
        print("Model initialized successfully")
    
    def prepare_datasets(self, texts, labels, test_size=0.2, random_state=42):
        """Prepare training and validation datasets"""
        print("Preparing datasets...")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Create datasets
        self.train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        self.val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        return train_texts, val_texts, train_labels, val_labels
    
    def train_model(self, epochs=3, batch_size=8, learning_rate=2e-5):
        """Train the BERT model"""
        print("Starting model training...")
        
        # Data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        best_val_f1 = 0.0
        training_stats = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            train_predictions = []
            train_true_labels = []
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
                # Store predictions for metrics
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                train_predictions.extend(preds.cpu().numpy())
                train_true_labels.extend(labels.cpu().numpy())
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    val_predictions.extend(preds.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_bert_model.pth')
                print(f"  New best model saved! F1: {val_f1:.4f}")
            
            # Store stats
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1,
                'val_loss': avg_val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_accuracy
            })
        
        print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")
        return training_stats
    
    def predict_test_data(self, test_texts, test_article_ids):
        """Generate predictions for test data"""
        print("Generating test predictions...")
        
        # Load best model
        if os.path.exists('best_bert_model.pth'):
            self.model.load_state_dict(torch.load('best_bert_model.pth', map_location=self.device))
            print("Loaded best model for predictions")
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        # Process in batches
        batch_size = 16
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            
            # Tokenize batch
            batch_encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = batch_encodings['input_ids'].to(self.device)
            attention_mask = batch_encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                batch_preds = torch.argmax(logits, dim=1)
                batch_probs = probs[:, 1].cpu().numpy()  # Probability of class 1 (real)
                
                predictions.extend(batch_preds.cpu().numpy())
                probabilities.extend(batch_probs)
        
        # Create submission dataframe
        submission_data = []
        for article_id, pred, prob in zip(test_article_ids, predictions, probabilities):
            # Convert prediction back to original format: 0->2 (fake), 1->1 (real)
            real_text_id = 1 if pred == 1 else 2
            submission_data.append({
                'id': article_id,
                'real_text_id': real_text_id,
                'confidence': prob if pred == 1 else 1 - prob
            })
        
        submission_df = pd.DataFrame(submission_data)
        return submission_df
    
    def run_full_pipeline(self):
        """Run the complete BERT fine-tuning pipeline"""
        print("=" * 60)
        print("BERT FINE-TUNING PIPELINE")
        print("=" * 60)
        
        try:
            # Load data
            train_texts, train_labels = self.load_text_data()
            
            if len(train_texts) == 0:
                print("No training data found!")
                return None
            
            # Initialize model
            self.initialize_model()
            
            # Prepare datasets
            self.prepare_datasets(train_texts, train_labels)
            
            # Train model
            training_stats = self.train_model(epochs=3, batch_size=8, learning_rate=2e-5)
            
            # Load test data and generate predictions
            test_texts, test_article_ids = self.load_test_data()
            submission_df = self.predict_test_data(test_texts, test_article_ids)
            
            # Save results
            submission_df.to_csv('submissions/bert_fine_tuned_submission.csv', index=False)
            print(f"\nSubmission saved to: submissions/bert_fine_tuned_submission.csv")
            print(f"Submission shape: {submission_df.shape}")
            
            # Save training stats
            with open('bert_training_stats.json', 'w') as f:
                json.dump(training_stats, f, indent=2)
            print("Training stats saved to: bert_training_stats.json")
            
            return submission_df
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    # Create submissions directory if it doesn't exist
    os.makedirs('submissions', exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = BERTFineTuningPipeline(
        model_name='bert-base-uncased',
        max_length=512
    )
    
    result = pipeline.run_full_pipeline()
    
    if result is not None:
        print("\n" + "=" * 60)
        print("BERT FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Generated {len(result)} predictions")
        print("Check submissions/bert_fine_tuned_submission.csv for results")
    else:
        print("\nPipeline failed. Check error messages above.")

if __name__ == "__main__":
    main()
