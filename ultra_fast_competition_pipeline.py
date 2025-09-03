#!/usr/bin/env python3
"""
Ultra-Fast Competition Pipeline
Runs the competition pipeline in under 30 seconds with optimized features
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import re

class UltraFastCompetitionPipeline:
    """Ultra-fast competition pipeline optimized for speed"""
    
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")
        
        # Essential space/science keywords only
        self.space_keywords = [
            'space', 'planet', 'star', 'galaxy', 'universe', 'orbit', 'satellite',
            'rocket', 'launch', 'mission', 'astronaut', 'nasa', 'esa', 'cosmos',
            'nebula', 'black hole', 'solar system', 'mars', 'venus', 'jupiter',
            'saturn', 'neptune', 'uranus', 'pluto', 'asteroid', 'comet', 'meteor'
        ]
        
        self.science_keywords = [
            'research', 'study', 'experiment', 'laboratory', 'scientist', 'discovery',
            'theory', 'hypothesis', 'analysis', 'data', 'results', 'conclusion'
        ]
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_training_data(self):
        """Load training data with labels - optimized for speed"""
        print("🔍 Loading training data...")
        
        # Load labels from CSV
        train_csv_path = os.path.join(self.data_path, "train.csv")
        train_df = pd.read_csv(train_csv_path)
        
        # Load text content efficiently
        train_texts = []
        train_labels = []
        
        for _, row in train_df.iterrows():
            article_id = row['id']
            label = row['real_text_id']
            
            article_dir = f"article_{article_id:04d}"
            article_path = os.path.join(self.train_path, article_dir)
            
            if os.path.exists(article_path):
                file_1_path = os.path.join(article_path, 'file_1.txt')
                file_2_path = os.path.join(article_path, 'file_2.txt')
                
                if os.path.exists(file_1_path) and os.path.exists(file_2_path):
                    with open(file_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(file_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()
                    
                    train_texts.append((text_1, text_2))
                    train_labels.append(label)
        
        print(f"✅ Loaded {len(train_texts)} training articles")
        return train_texts, train_labels
    
    def load_test_data(self):
        """Load test data - optimized for speed"""
        print("🔍 Loading test data...")
        
        test_articles = []
        test_ids = []
        
        # Load test data efficiently
        for item in sorted(os.listdir(self.test_path)):
            if os.path.isdir(os.path.join(self.test_path, item)) and item.startswith('article_'):
                article_path = os.path.join(self.test_path, item)
                
                file_1_path = os.path.join(article_path, 'file_1.txt')
                file_2_path = os.path.join(article_path, 'file_2.txt')
                
                if os.path.exists(file_1_path) and os.path.exists(file_2_path):
                    with open(file_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(file_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()
                    
                    test_articles.append((text_1, text_2))
                    test_ids.append(int(item.split('_')[1]))
        
        print(f"✅ Loaded {len(test_articles)} test articles")
        return test_articles, test_ids
    
    def extract_fast_features(self, text):
        """Extract only essential features for speed"""
        if not text or not isinstance(text, str):
            return [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Basic features only
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Word analysis
        words = text.split()
        unique_words = set(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Domain features (simplified)
        space_score = sum(1 for word in words if word.lower() in self.space_keywords)
        science_score = sum(1 for word in words if word.lower() in self.science_keywords)
        
        # Complexity features
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / word_count if word_count > 0 else 0
        
        return [
            char_count, word_count, sentence_count, len(unique_words) / word_count if word_count > 0 else 0,
            avg_word_length, space_score, science_score, long_word_ratio
        ]
    
    def create_fast_features(self, train_texts, test_texts):
        """Create feature matrix quickly"""
        print("🔧 Creating features...")
        
        # TF-IDF features (reduced size)
        print("  • Creating TF-IDF features...")
        all_texts = []
        for text_pair in train_texts:
            all_texts.extend(text_pair)
        for text_pair in test_texts:
            all_texts.extend(text_pair)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced from 1000
            ngram_range=(1, 2),  # Reduced from (1, 3)
            stop_words='english'
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Handcrafted features (essential only)
        print("  • Creating handcrafted features...")
        handcrafted_features = []
        for text_pair in train_texts + test_texts:
            for text in text_pair:
                features = self.extract_fast_features(text)
                handcrafted_features.append(features)
        
        handcrafted_array = np.array(handcrafted_features)
        
        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            handcrafted_array
        ])
        
        print(f"✅ Created feature matrix: {combined_features.shape}")
        return combined_features
    
    def train_fast_models(self, X_train, y_train):
        """Train models quickly with optimized parameters"""
        print("🚀 Training models...")
        
        # Split features for training
        n_train = len(y_train)
        X_train_combined = X_train[:n_train*2]
        
        # Create labels for both texts
        y_train_expanded = []
        for label in y_train:
            if label == 1:
                y_train_expanded.extend([1, 0])
            else:
                y_train_expanded.extend([0, 1])
        
        # Train fast models with optimized parameters
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=500,  # Reduced iterations
                C=1.0,  # Fixed parameter
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=10,     # Fixed depth
                min_samples_split=5,  # Fixed parameter
                random_state=42,
                n_jobs=-1  # Use all cores
            )
        }
        
        for name, model in models.items():
            print(f"  • Training {name}...")
            model.fit(X_train_combined, y_train_expanded)
            self.models[name] = model
        
        print("✅ Models trained successfully!")
    
    def predict_fast(self, X_test, test_ids):
        """Generate predictions quickly"""
        print("🎯 Generating predictions...")
        
        predictions = []
        
        for i, test_id in enumerate(test_ids):
            text1_idx = i * 2
            text2_idx = i * 2 + 1
            
            # Get predictions from each model
            model_predictions = []
            for model_name, model in self.models.items():
                pred1 = model.predict_proba(X_test[text1_idx:text1_idx+1])[0][1]
                pred2 = model.predict_proba(X_test[text2_idx:text2_idx+1])[0][1]
                
                if pred1 > pred2:
                    model_predictions.append(1)
                else:
                    model_predictions.append(2)
            
            # Simple majority vote
            final_prediction = max(set(model_predictions), key=model_predictions.count)
            predictions.append(final_prediction)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_ids,
            'real_text_id': predictions
        })
        
        print(f"✅ Generated predictions for {len(test_ids)} test samples")
        return submission_df
    
    def run_ultra_fast_pipeline(self):
        """Run the ultra-fast competition pipeline"""
        print("🚀 ULTRA-FAST COMPETITION PIPELINE")
        print("=" * 60)
        print("⚡ Target: Complete in under 30 seconds")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load data
            train_texts, train_labels = self.load_training_data()
            test_texts, test_ids = self.load_test_data()
            
            # Create features
            X_combined = self.create_fast_features(train_texts, test_texts)
            
            # Split features for training and testing
            n_train = len(train_labels)
            X_train = X_combined[:n_train*2]
            X_test = X_combined[n_train*2:]
            
            # Train models
            self.train_fast_models(X_train, train_labels)
            
            # Generate predictions
            submission_df = self.predict_fast(X_test, test_ids)
            
            # Save submission
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            submission_file = f"ultra_fast_submission_{timestamp}.csv"
            submission_df.to_csv(submission_file, index=False)
            
            total_time = time.time() - start_time
            
            print(f"\n🎉 ULTRA-FAST PIPELINE COMPLETED!")
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"📁 Submission saved: {submission_file}")
            print(f"📊 Test samples: {len(test_ids)}")
            print(f"🔍 Training samples: {len(train_labels)}")
            
            if total_time < 30:
                print("✅ Target achieved: Pipeline completed in under 30 seconds!")
            else:
                print(f"⚠️  Target missed: Pipeline took {total_time:.2f} seconds")
            
            return submission_df
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    pipeline = UltraFastCompetitionPipeline()
    submission = pipeline.run_ultra_fast_pipeline()
