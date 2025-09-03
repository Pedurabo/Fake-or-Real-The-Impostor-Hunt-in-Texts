#!/usr/bin/env python3
"""
BERT Ensemble Pipeline
Combines multiple BERT models and approaches for improved performance
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

class EnsembleTextDataset(Dataset):
    """Dataset for ensemble training"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
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

class BERTEnsemblePipeline:
    """Ensemble pipeline combining multiple BERT models"""
    
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        self.predictions = {}
        
        print(f"Using device: {self.device}")
        print(f"Max sequence length: {max_length}")
    
    def load_text_data(self):
        """Load and preprocess text data"""
        print("Loading text data...")
        
        train_labels = pd.read_csv('data/train.csv')
        print(f"Training labels shape: {train_labels.shape}")
        
        train_texts = []
        train_labels_list = []
        
        for idx, row in train_labels.iterrows():
            article_id = row['id']
            real_text_id = row['real_text_id']
            
            article_dir = f"article_{article_id:04d}"
            article_path = os.path.join('data/train', article_dir)
            
            if os.path.exists(article_path):
                article_texts = []
                for file_name in os.listdir(article_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(article_path, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read().strip()
                                if text and len(text) > 50:
                                    article_texts.append(text)
                        except:
                            continue
                
                if article_texts:
                    combined_text = ' [SEP] '.join(article_texts)
                    if len(combined_text) > self.max_length * 4:
                        combined_text = combined_text[:self.max_length * 4]
                    
                    train_texts.append(combined_text)
                    label = 1 if real_text_id == 1 else 0
                    train_labels_list.append(label)
        
        print(f"Loaded {len(train_texts)} training samples")
        print(f"Label distribution: {np.bincount(train_labels_list)}")
        
        return train_texts, train_labels_list
    
    def load_test_data(self):
        """Load test data"""
        print("Loading test data...")
        
        test_texts = []
        test_article_ids = []
        
        test_dir = 'data/test'
        for article_dir in sorted(os.listdir(test_dir)):
            if article_dir.startswith('article_'):
                article_path = os.path.join(test_dir, article_dir)
                article_id = int(article_dir.replace('article_', ''))
                
                article_texts = []
                for file_name in os.listdir(article_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(article_path, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read().strip()
                                if text and len(text) > 50:
                                    article_texts.append(text)
                        except:
                            continue
                
                if article_texts:
                    combined_text = ' [SEP] '.join(article_texts)
                    if len(combined_text) > self.max_length * 4:
                        combined_text = combined_text[:self.max_length * 4]
                    
                    test_texts.append(combined_text)
                    test_article_ids.append(article_id)
        
        print(f"Loaded {len(test_texts)} test samples")
        return test_texts, test_article_ids
    
    def initialize_models(self):
        """Initialize multiple BERT models"""
        print("Initializing ensemble models...")
        
        # Model 1: BERT Base
        print("  Initializing BERT Base...")
        self.tokenizers['bert_base'] = BertTokenizer.from_pretrained('bert-base-uncased')
        self.models['bert_base'] = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2
        ).to(self.device)
        
        # Model 2: DistilBERT (faster, smaller)
        print("  Initializing DistilBERT...")
        self.tokenizers['distilbert'] = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.models['distilbert'] = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=2
        ).to(self.device)
        
        # Model 3: BERT with different max length
        print("  Initializing BERT (shorter sequences)...")
        self.tokenizers['bert_short'] = BertTokenizer.from_pretrained('bert-base-uncased')
        self.models['bert_short'] = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2
        ).to(self.device)
        
        print("All models initialized successfully")
    
    def train_single_model(self, model_name, train_texts, train_labels, val_texts, val_labels, 
                          epochs=2, batch_size=16, learning_rate=3e-5):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Create datasets
        train_dataset = EnsembleTextDataset(train_texts, train_labels, tokenizer, self.max_length)
        val_dataset = EnsembleTextDataset(val_texts, val_labels, tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps
        )
        
        best_val_f1 = 0.0
        training_stats = []
        
        for epoch in range(epochs):
            print(f"  Epoch {epoch + 1}/{epochs}")
            
            # Training
            model.train()
            total_train_loss = 0
            train_predictions = []
            train_true_labels = []
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                model.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                train_predictions.extend(preds.cpu().numpy())
                train_true_labels.extend(labels.cpu().numpy())
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
            
            # Validation
            model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    val_predictions.extend(preds.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            
            print(f"    Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"    Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'best_{model_name}_model.pth')
                print(f"    New best model saved! F1: {val_f1:.4f}")
            
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1,
                'val_loss': avg_val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_accuracy
            })
        
        print(f"  {model_name} training completed! Best validation F1: {best_val_f1:.4f}")
        return training_stats
    
    def train_ensemble_models(self, train_texts, train_labels):
        """Train all ensemble models using cross-validation"""
        print("Training ensemble models...")
        
        # Use stratified k-fold for better validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        all_training_stats = {}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
            print(f"\nFold {fold + 1}/3")
            
            fold_train_texts = [train_texts[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_texts = [train_texts[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]
            
            # Train each model on this fold
            for model_name in self.models.keys():
                if model_name not in all_training_stats:
                    all_training_stats[model_name] = []
                
                stats = self.train_single_model(
                    model_name, fold_train_texts, fold_train_labels,
                    fold_val_texts, fold_val_labels,
                    epochs=2, batch_size=16, learning_rate=3e-5
                )
                
                all_training_stats[model_name].extend(stats)
        
        # Save training stats
        with open('ensemble_training_stats.json', 'w') as f:
            json.dump(all_training_stats, f, indent=2)
        print("Training stats saved to: ensemble_training_stats.json")
        
        return all_training_stats
    
    def predict_with_single_model(self, model_name, test_texts, test_article_ids):
        """Generate predictions with a single model"""
        print(f"Generating predictions with {model_name}...")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Load best model if available
        model_path = f'best_{model_name}_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"  Loaded best {model_name} model")
        
        model.eval()
        predictions = []
        probabilities = []
        
        batch_size = 32
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            
            batch_encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = batch_encodings['input_ids'].to(self.device)
            attention_mask = batch_encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                batch_preds = torch.argmax(logits, dim=1)
                batch_probs = probs[:, 1].cpu().numpy()
                
                predictions.extend(batch_preds.cpu().numpy())
                probabilities.extend(batch_probs)
        
        return predictions, probabilities
    
    def create_ensemble_predictions(self, test_texts, test_article_ids):
        """Create ensemble predictions from all models"""
        print("Creating ensemble predictions...")
        
        all_predictions = {}
        all_probabilities = {}
        
        # Get predictions from each model
        for model_name in self.models.keys():
            print(f"\nGetting predictions from {model_name}...")
            predictions, probabilities = self.predict_with_single_model(
                model_name, test_texts, test_article_ids
            )
            all_predictions[model_name] = predictions
            all_probabilities[model_name] = probabilities
        
        # Create ensemble predictions
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for i in range(len(test_texts)):
            # Collect predictions from all models
            model_preds = [all_predictions[model_name][i] for model_name in self.models.keys()]
            model_probs = [all_probabilities[model_name][i] for model_name in self.models.keys()]
            
            # Majority voting for final prediction
            final_pred = 1 if sum(model_preds) > len(model_preds) / 2 else 0
            
            # Average probabilities for confidence
            avg_prob = np.mean(model_probs)
            
            ensemble_predictions.append(final_pred)
            ensemble_probabilities.append(avg_prob)
        
        # Create submission dataframe
        submission_data = []
        for article_id, pred, prob in zip(test_article_ids, ensemble_predictions, ensemble_probabilities):
            real_text_id = 1 if pred == 1 else 2
            submission_data.append({
                'id': article_id,
                'real_text_id': real_text_id,
                'confidence': prob if pred == 1 else 1 - prob
            })
        
        submission_df = pd.DataFrame(submission_data)
        
        # Save individual model predictions for analysis
        for model_name in self.models.keys():
            model_submission = []
            for article_id, pred, prob in zip(test_article_ids, all_predictions[model_name], all_probabilities[model_name]):
                real_text_id = 1 if pred == 1 else 2
                model_submission.append({
                    'id': article_id,
                    'real_text_id': real_text_id,
                    'confidence': prob if pred == 1 else 1 - prob
                })
            
            model_df = pd.DataFrame(model_submission)
            model_df.to_csv(f'submissions/{model_name}_submission.csv', index=False)
            print(f"  {model_name} submission saved")
        
        return submission_df
    
    def run_ensemble_pipeline(self):
        """Run the complete ensemble pipeline"""
        print("=" * 60)
        print("BERT ENSEMBLE PIPELINE")
        print("=" * 60)
        
        try:
            # Load data
            train_texts, train_labels = self.load_text_data()
            
            if len(train_texts) == 0:
                print("No training data found!")
                return None
            
            # Initialize models
            self.initialize_models()
            
            # Train ensemble models
            training_stats = self.train_ensemble_models(train_texts, train_labels)
            
            # Generate ensemble predictions
            test_texts, test_article_ids = self.load_test_data()
            submission_df = self.create_ensemble_predictions(test_texts, test_article_ids)
            
            # Save ensemble results
            submission_df.to_csv('submissions/bert_ensemble_submission.csv', index=False)
            print(f"\nEnsemble submission saved to: submissions/bert_ensemble_submission.csv")
            print(f"Ensemble submission shape: {submission_df.shape}")
            
            return submission_df
            
        except Exception as e:
            print(f"Error in ensemble pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    os.makedirs('submissions', exist_ok=True)
    
    # Initialize and run ensemble pipeline
    pipeline = BERTEnsemblePipeline(max_length=256)
    
    result = pipeline.run_ensemble_pipeline()
    
    if result is not None:
        print("\n" + "=" * 60)
        print("BERT ENSEMBLE PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Generated {len(result)} ensemble predictions")
        print("Check submissions/bert_ensemble_submission.csv for results")
        print("Individual model submissions also saved")
    else:
        print("\nEnsemble pipeline failed. Check error messages above.")

if __name__ == "__main__":
    main()
