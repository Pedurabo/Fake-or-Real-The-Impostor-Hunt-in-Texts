#!/usr/bin/env python3
"""
Super-Fast Competition Predictions
Loads pre-trained models and generates predictions in seconds
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
import pickle
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
import re

class SuperFastPredictor:
    """Super-fast predictor using pre-trained models"""
    
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.test_path = os.path.join(data_path, "test")
        
        # Load pre-trained models and vectorizers
        self.models = {}
        self.vectorizer = None
        self.load_pretrained_models()
        
    def load_pretrained_models(self):
        """Load pre-trained models for instant predictions"""
        print("🔍 Loading pre-trained models...")
        
        # Try to load from existing files
        model_files = [
            "production_models/best_model.pkl",
            "models/best_model.pkl", 
            "best_model.pkl"
        ]
        
        vectorizer_files = [
            "production_models/vectorizer.pkl",
            "models/vectorizer.pkl",
            "vectorizer.pkl"
        ]
        
        # Load model
        model_loaded = False
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        self.models['main'] = pickle.load(f)
                    print(f"✅ Loaded model from {model_file}")
                    model_loaded = True
                    break
                except:
                    continue
        
        # Load vectorizer
        vectorizer_loaded = False
        for vec_file in vectorizer_files:
            if os.path.exists(vec_file):
                try:
                    with open(vec_file, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    print(f"✅ Loaded vectorizer from {vec_file}")
                    vectorizer_loaded = True
                    break
                except:
                    continue
        
        if not model_loaded or not vectorizer_loaded:
            print("⚠️  Pre-trained models not found. Using fallback approach...")
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create simple fallback models if pre-trained ones aren't available"""
        print("🔧 Creating fallback models...")
        
        # Simple TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=200,  # Very small for speed
            ngram_range=(1, 1),  # Only unigrams
            stop_words='english'
        )
        
        # Simple model (will be trained quickly)
        from sklearn.linear_model import LogisticRegression
        self.models['main'] = LogisticRegression(random_state=42, max_iter=100)
        
        print("✅ Fallback models created")
    
    def load_test_data_fast(self):
        """Load test data as quickly as possible"""
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
    
    def extract_minimal_features(self, text):
        """Extract only the most essential features for speed"""
        if not text or not isinstance(text, str):
            return [0, 0, 0, 0]
        
        # Ultra-minimal features
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # Simple complexity measure
        long_words = sum(1 for w in text.split() if len(w) > 6)
        long_word_ratio = long_words / word_count if word_count > 0 else 0
        
        return [char_count, word_count, avg_word_length, long_word_ratio]
    
    def create_minimal_features(self, test_texts):
        """Create minimal feature matrix for speed"""
        print("🔧 Creating minimal features...")
        
        # TF-IDF features only
        all_texts = []
        for text_pair in test_texts:
            all_texts.extend(text_pair)
        
        # Use existing vectorizer or fit new one
        if hasattr(self.vectorizer, 'vocabulary_'):
            # Vectorizer already fitted
            tfidf_features = self.vectorizer.transform(all_texts)
        else:
            # Need to fit vectorizer (this will be fast with small features)
            tfidf_features = self.vectorizer.fit_transform(all_texts)
        
        # Minimal handcrafted features
        handcrafted_features = []
        for text_pair in test_texts:
            for text in text_pair:
                features = self.extract_minimal_features(text)
                handcrafted_features.append(features)
        
        handcrafted_array = np.array(handcrafted_features)
        
        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            handcrafted_array
        ])
        
        print(f"✅ Created minimal feature matrix: {combined_features.shape}")
        return combined_features
    
    def predict_super_fast(self, X_test, test_ids):
        """Generate predictions super fast"""
        print("🎯 Generating predictions...")
        
        predictions = []
        
        # Use the main model for predictions
        model = self.models['main']
        
        for i, test_id in enumerate(test_ids):
            text1_idx = i * 2
            text2_idx = i * 2 + 1
            
            # Get predictions for both texts
            pred1 = model.predict_proba(X_test[text1_idx:text1_idx+1])[0][1]
            pred2 = model.predict_proba(X_test[text2_idx:text2_idx+1])[0][1]
            
            # Choose the text with higher probability of being real
            if pred1 > pred2:
                predictions.append(1)
            else:
                predictions.append(2)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_ids,
            'real_text_id': predictions
        })
        
        print(f"✅ Generated predictions for {len(test_ids)} test samples")
        return submission_df
    
    def run_super_fast_pipeline(self):
        """Run the super-fast prediction pipeline"""
        print("🚀 SUPER-FAST COMPETITION PREDICTIONS")
        print("=" * 60)
        print("⚡ Target: Complete in under 10 seconds")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load test data
            test_texts, test_ids = self.load_test_data_fast()
            
            # Create minimal features
            X_test = self.create_minimal_features(test_texts)
            
            # Generate predictions
            submission_df = self.predict_super_fast(X_test, test_ids)
            
            # Save submission
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            submission_file = f"super_fast_submission_{timestamp}.csv"
            submission_df.to_csv(submission_file, index=False)
            
            total_time = time.time() - start_time
            
            print(f"\n🎉 SUPER-FAST PIPELINE COMPLETED!")
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"📁 Submission saved: {submission_file}")
            print(f"📊 Test samples: {len(test_ids)}")
            
            if total_time < 10:
                print("✅ Target achieved: Pipeline completed in under 10 seconds!")
            elif total_time < 30:
                print("✅ Good: Pipeline completed in under 30 seconds!")
            else:
                print(f"⚠️  Target missed: Pipeline took {total_time:.2f} seconds")
            
            return submission_df
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    predictor = SuperFastPredictor()
    submission = predictor.run_super_fast_pipeline()
