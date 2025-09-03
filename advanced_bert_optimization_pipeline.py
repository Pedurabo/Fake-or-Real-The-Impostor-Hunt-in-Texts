#!/usr/bin/env python3
"""
Advanced BERT Optimization Pipeline
Implements multiple BERT optimization techniques to improve beyond 0.73858
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class AdvancedBERTOptimizer:
    """Advanced BERT optimization with multiple techniques"""
    
    def __init__(self, max_length=256, batch_size=16):
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        self.optimization_results = {}
        
        print(f"Advanced BERT Optimizer initialized")
        print(f"Device: {self.device}")
        print(f"Max length: {max_length}")
        print(f"Batch size: {batch_size}")
    
    def load_multiple_bert_models(self):
        """Load multiple BERT variants for ensemble"""
        print("\n" + "="*60)
        print("LOADING MULTIPLE BERT VARIANTS")
        print("="*60)
        
        models_to_load = {
            'bert_base': {
                'tokenizer': 'bert-base-uncased',
                'model': 'bert-base-uncased'
            },
            'distilbert': {
                'tokenizer': 'distilbert-base-uncased',
                'model': 'distilbert-base-uncased'
            },
            'roberta': {
                'tokenizer': 'roberta-base',
                'model': 'roberta-base'
            }
        }
        
        for name, config in models_to_load.items():
            try:
                print(f"Loading {name}...")
                
                # Load tokenizer
                if 'roberta' in name:
                    tokenizer_class = RobertaTokenizer
                elif 'distilbert' in name:
                    tokenizer_class = DistilBertTokenizer
                else:
                    tokenizer_class = BertTokenizer
                
                self.tokenizers[name] = tokenizer_class.from_pretrained(config['tokenizer'])
                
                # Load model
                if 'roberta' in name:
                    model_class = RobertaForSequenceClassification
                elif 'distilbert' in name:
                    model_class = DistilBertForSequenceClassification
                else:
                    model_class = BertForSequenceClassification
                
                self.models[name] = model_class.from_pretrained(config['model'])
                self.models[name].to(self.device)
                self.models[name].eval()
                
                print(f"✓ {name} loaded successfully")
                
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        return len(self.models) > 0
    
    def extract_advanced_features(self, texts, model_name):
        """Extract advanced features from specific BERT model"""
        print(f"Extracting features using {model_name}...")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        all_features = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize batch
            batch_encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = batch_encodings['input_ids'].to(self.device)
            attention_mask = batch_encodings['attention_mask'].to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Multiple feature extraction strategies
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Use multiple layers for richer features
                    layer_features = []
                    for layer_idx in [0, 6, 12]:  # First, middle, last layers
                        if layer_idx < len(outputs.hidden_states):
                            layer_output = outputs.hidden_states[layer_idx]
                            # CLS token + mean pooling
                            cls_feat = layer_output[:, 0, :].cpu().numpy()
                            mean_feat = layer_output.mean(dim=1).cpu().numpy()
                            layer_features.append(np.concatenate([cls_feat, mean_feat], axis=1))
                    
                    if layer_features:
                        # Concatenate all layer features
                        combined_features = np.concatenate(layer_features, axis=1)
                        all_features.extend(combined_features)
                    else:
                        # Fallback to last hidden state
                        last_hidden = outputs.last_hidden_state
                        cls_features = last_hidden[:, 0, :].cpu().numpy()
                        all_features.extend(cls_features)
                else:
                    # Standard approach
                    last_hidden = outputs.last_hidden_state
                    cls_features = last_hidden[:, 0, :].cpu().numpy()
                    all_features.extend(cls_features)
        
        return np.array(all_features)
    
    def create_advanced_classifiers(self):
        """Create multiple advanced classifiers"""
        print("\n" + "="*60)
        print("CREATING ADVANCED CLASSIFIERS")
        print("="*60)
        
        classifiers = {}
        
        # 1. Optimized Logistic Regression
        print("Creating optimized Logistic Regression...")
        lr_params = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'solver': ['liblinear', 'saga']
        }
        
        lr_base = LogisticRegression(random_state=42)
        lr_optimized = GridSearchCV(lr_base, lr_params, cv=3, scoring='f1_weighted', n_jobs=-1)
        classifiers['logistic_regression'] = lr_optimized
        
        # 2. Random Forest with optimization
        print("Creating optimized Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_optimized = GridSearchCV(rf_base, rf_params, cv=3, scoring='f1_weighted', n_jobs=-1)
        classifiers['random_forest'] = rf_optimized
        
        # 3. Voting Classifier
        print("Creating Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=42, max_iter=2000)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=200))
            ],
            voting='soft'
        )
        classifiers['voting'] = voting_clf
        
        print(f"✓ Created {len(classifiers)} advanced classifiers")
        return classifiers
    
    def cross_validate_bert_features(self, features, labels, n_splits=5):
        """Cross-validate BERT features for robust performance"""
        print(f"\nCross-validating with {n_splits} folds...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            print(f"  Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Train and evaluate multiple classifiers
            classifiers = self.create_advanced_classifiers()
            
            fold_results = {}
            for name, clf in classifiers.items():
                try:
                    # Train classifier
                    clf.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = clf.predict(X_val)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    
                    fold_results[name] = {
                        'accuracy': accuracy,
                        'f1': f1,
                        'classifier': clf
                    }
                    
                    print(f"    {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                    
                except Exception as e:
                    print(f"    ✗ Error with {name}: {e}")
            
            # Find best classifier for this fold
            if fold_results:
                best_classifier = max(fold_results.items(), key=lambda x: x[1]['f1'])
                fold_scores.append({
                    'fold': fold + 1,
                    'best_classifier': best_classifier[0],
                    'best_f1': best_classifier[1]['f1'],
                    'best_accuracy': best_classifier[1]['accuracy'],
                    'all_results': fold_results
                })
        
        return fold_scores
    
    def optimize_bert_hyperparameters(self, texts, labels):
        """Optimize BERT hyperparameters for best performance"""
        print("\n" + "="*60)
        print("OPTIMIZING BERT HYPERPARAMETERS")
        print("="*60)
        
        # Test different max_length values
        max_lengths = [128, 256, 384, 512]
        batch_sizes = [8, 16, 32]
        
        optimization_results = {}
        
        for max_len in max_lengths:
            for batch_size in batch_sizes:
                print(f"\nTesting max_length={max_len}, batch_size={batch_size}")
                
                try:
                    # Update parameters
                    self.max_length = max_len
                    self.batch_size = batch_size
                    
                    # Extract features with current settings
                    features = self.extract_advanced_features(texts, 'bert_base')
                    
                    # Cross-validate
                    cv_results = self.cross_validate_bert_features(features, labels, n_splits=3)
                    
                    if cv_results:
                        avg_f1 = np.mean([result['best_f1'] for result in cv_results])
                        avg_accuracy = np.mean([result['best_accuracy'] for result in cv_results])
                        
                        optimization_results[f"max_len_{max_len}_batch_{batch_size}"] = {
                            'max_length': max_len,
                            'batch_size': batch_size,
                            'avg_f1': avg_f1,
                            'avg_accuracy': avg_accuracy,
                            'cv_results': cv_results
                        }
                        
                        print(f"  Results: F1={avg_f1:.4f}, Accuracy={avg_accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        # Find best configuration
        if optimization_results:
            best_config = max(optimization_results.items(), key=lambda x: x[1]['avg_f1'])
            print(f"\n✓ Best configuration: {best_config[0]}")
            print(f"  F1: {best_config[1]['avg_f1']:.4f}")
            print(f"  Accuracy: {best_config[1]['avg_accuracy']:.4f}")
            
            # Update to best configuration
            self.max_length = best_config[1]['max_length']
            self.batch_size = best_config[1]['batch_size']
        
        return optimization_results
    
    def create_bert_ensemble(self, features_dict, labels):
        """Create ensemble of multiple BERT models"""
        print("\n" + "="*60)
        print("CREATING BERT ENSEMBLE")
        print("="*60)
        
        ensemble_features = []
        ensemble_names = []
        
        # Combine features from all models
        for model_name, features in features_dict.items():
            ensemble_features.append(features)
            ensemble_names.append(model_name)
            print(f"Added {model_name}: {features.shape}")
        
        # Concatenate all features
        combined_features = np.concatenate(ensemble_features, axis=1)
        print(f"Combined features shape: {combined_features.shape}")
        
        # Train ensemble classifier
        print("Training ensemble classifier...")
        
        # Use Voting Classifier with multiple base classifiers
        base_classifiers = [
            ('lr', LogisticRegression(random_state=42, max_iter=2000)),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=200)),
            ('lr2', LogisticRegression(random_state=42, max_iter=2000, C=0.1))
        ]
        
        ensemble_clf = VotingClassifier(
            estimators=base_classifiers,
            voting='soft'
        )
        
        # Train ensemble
        ensemble_clf.fit(combined_features, labels)
        
        print("✓ BERT ensemble created successfully")
        return ensemble_clf, combined_features
    
    def run_advanced_optimization(self):
        """Run the complete advanced BERT optimization pipeline"""
        print("="*80)
        print("ADVANCED BERT OPTIMIZATION PIPELINE")
        print("="*80)
        
        # Step 1: Load multiple BERT models
        if not self.load_multiple_bert_models():
            print("Failed to load BERT models. Cannot proceed.")
            return None
        
        # Step 2: Load training data
        print("\nLoading training data...")
        try:
            train_labels = pd.read_csv('data/train.csv')
            print(f"Training labels shape: {train_labels.shape}")
            
            # Load training texts
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
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None
        
        # Step 3: Extract features from all models
        print("\nExtracting features from all BERT models...")
        features_dict = {}
        
        for model_name in self.models.keys():
            try:
                features = self.extract_advanced_features(train_texts, model_name)
                features_dict[model_name] = features
                print(f"✓ {model_name}: {features.shape}")
            except Exception as e:
                print(f"✗ Error extracting features from {model_name}: {e}")
        
        # Step 4: Optimize hyperparameters
        print("\nOptimizing BERT hyperparameters...")
        optimization_results = self.optimize_bert_hyperparameters(train_texts, train_labels_list)
        
        # Step 5: Create BERT ensemble
        print("\nCreating BERT ensemble...")
        ensemble_clf, combined_features = self.create_bert_ensemble(features_dict, train_labels_list)
        
        # Step 6: Cross-validate ensemble
        print("\nCross-validating ensemble...")
        cv_results = self.cross_validate_bert_features(combined_features, train_labels_list, n_splits=5)
        
        # Step 7: Generate optimization report
        self.generate_optimization_report(optimization_results, cv_results, features_dict)
        
        return {
            'ensemble_classifier': ensemble_clf,
            'combined_features': combined_features,
            'features_dict': features_dict,
            'optimization_results': optimization_results,
            'cv_results': cv_results
        }
    
    def generate_optimization_report(self, optimization_results, cv_results, features_dict):
        """Generate comprehensive optimization report"""
        print("\n" + "="*60)
        print("GENERATING OPTIMIZATION REPORT")
        print("="*60)
        
        # Create report data
        report_data = {
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'current_baseline_score': 0.73858,
            'optimization_summary': {},
            'feature_dimensions': {},
            'best_configurations': {},
            'cross_validation_results': {}
        }
        
        # Add feature dimensions
        for model_name, features in features_dict.items():
            report_data['feature_dimensions'][model_name] = {
                'shape': features.shape,
                'total_features': features.shape[1]
            }
        
        # Add optimization results
        if optimization_results:
            best_config = max(optimization_results.items(), key=lambda x: x[1]['avg_f1'])
            report_data['best_configurations'] = {
                'best_config': best_config[0],
                'max_length': best_config[1]['max_length'],
                'batch_size': best_config[1]['batch_size'],
                'best_f1': best_config[1]['avg_f1'],
                'best_accuracy': best_config[1]['avg_accuracy']
            }
        
        # Add cross-validation results
        if cv_results:
            avg_f1 = np.mean([result['best_f1'] for result in cv_results])
            avg_accuracy = np.mean([result['best_accuracy'] for result in cv_results])
            
            report_data['cross_validation_results'] = {
                'average_f1': avg_f1,
                'average_accuracy': avg_accuracy,
                'fold_results': cv_results
            }
        
        # Save report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'advanced_bert_optimization_report_{timestamp}.json'
        
        with open(report_filename, 'w') as f:
            import json
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✓ Optimization report saved: {report_filename}")
        
        # Print summary
        print("\nOPTIMIZATION SUMMARY:")
        print("-" * 40)
        print(f"Current Baseline Score: 0.73858")
        
        if 'best_configurations' in report_data and report_data['best_configurations']:
            best = report_data['best_configurations']
            print(f"Best Configuration: {best['best_config']}")
            print(f"Best F1: {best['best_f1']:.4f}")
            print(f"Best Accuracy: {best['best_accuracy']:.4f}")
        
        if 'cross_validation_results' in report_data and report_data['cross_validation_results']:
            cv = report_data['cross_validation_results']
            print(f"Ensemble CV F1: {cv['average_f1']:.4f}")
            print(f"Ensemble CV Accuracy: {cv['average_accuracy']:.4f}")
        
        return report_data

def main():
    """Main execution function"""
    print("Starting Advanced BERT Optimization Pipeline...")
    
    # Initialize optimizer
    optimizer = AdvancedBERTOptimizer(max_length=256, batch_size=16)
    
    # Run optimization
    results = optimizer.run_advanced_optimization()
    
    if results:
        print("\n" + "="*80)
        print("ADVANCED BERT OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Ready to generate improved competition submission!")
    else:
        print("\nOptimization failed. Check error messages above.")

if __name__ == "__main__":
    main()
