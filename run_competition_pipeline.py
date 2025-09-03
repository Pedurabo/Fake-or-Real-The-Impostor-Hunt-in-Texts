#!/usr/bin/env python3
"""
Competition Pipeline Runner
Runs the best performing pipeline on the actual competition data
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import re
from collections import Counter
import math

class CompetitionPipelineRunner:
    """Streamlined competition pipeline runner"""
    
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")
        
        # Space/science domain keywords
        self.space_keywords = [
            'space', 'planet', 'star', 'galaxy', 'universe', 'orbit', 'satellite',
            'rocket', 'launch', 'mission', 'astronaut', 'nasa', 'esa', 'cosmos',
            'nebula', 'black hole', 'solar system', 'mars', 'venus', 'jupiter',
            'saturn', 'neptune', 'uranus', 'pluto', 'asteroid', 'comet', 'meteor',
            'gravity', 'atmosphere', 'oxygen', 'hydrogen', 'helium', 'carbon',
            'radiation', 'vacuum', 'zero gravity', 'microgravity', 'space station',
            'spacecraft', 'rover', 'probe', 'telescope', 'observatory', 'hubble',
            'iss', 'international space station', 'apollo', 'soyuz', 'spacex'
        ]
        
        self.science_keywords = [
            'research', 'study', 'experiment', 'laboratory', 'scientist', 'discovery',
            'theory', 'hypothesis', 'analysis', 'data', 'results', 'conclusion',
            'peer review', 'publication', 'journal', 'conference', 'methodology',
            'sample', 'control', 'variable', 'statistics', 'correlation', 'causation'
        ]
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_training_data(self):
        """Load training data with labels"""
        print("🔍 Loading training data...")
        
        # Load labels from CSV
        train_csv_path = os.path.join(self.data_path, "train.csv")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")
        
        train_df = pd.read_csv(train_csv_path)
        print(f"✅ Loaded {len(train_df)} training labels from CSV")
        
        # Load text content
        train_texts = []
        train_labels = []
        
        for _, row in train_df.iterrows():
            article_id = row['id']
            label = row['real_text_id']
            
            # Find article directory
            article_dir = f"article_{article_id:04d}"
            article_path = os.path.join(self.train_path, article_dir)
            
            if os.path.exists(article_path):
                # Load both text files
                file_1_path = os.path.join(article_path, 'file_1.txt')
                file_2_path = os.path.join(article_path, 'file_2.txt')
                
                if os.path.exists(file_1_path) and os.path.exists(file_2_path):
                    with open(file_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(file_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()
                    
                    # Create features for both texts
                    features_1 = self.extract_text_features(text_1)
                    features_2 = self.extract_text_features(text_2)
                    
                    # Store both with their labels
                    train_texts.append((text_1, text_2))
                    train_labels.append(label)
        
        print(f"✅ Loaded {len(train_texts)} training articles with labels")
        return train_texts, train_labels
    
    def load_test_data(self):
        """Load test data"""
        print("🔍 Loading test data...")
        
        test_articles = []
        test_ids = []
        
        # Find all test article directories
        for item in sorted(os.listdir(self.test_path)):
            if os.path.isdir(os.path.join(self.test_path, item)) and item.startswith('article_'):
                article_path = os.path.join(self.test_path, item)
                
                # Load both text files
                file_1_path = os.path.join(article_path, 'file_1.txt')
                file_2_path = os.path.join(article_path, 'file_2.txt')
                
                if os.path.exists(file_1_path) and os.path.exists(file_2_path):
                    with open(file_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(file_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()
                    
                    test_articles.append((text_1, text_2))
                    test_ids.append(int(item.split('_')[1]))  # Extract ID from directory name
        
        print(f"✅ Loaded {len(test_articles)} test articles")
        return test_articles, test_ids
    
    def extract_text_features(self, text):
        """Extract comprehensive text features"""
        if not text or not isinstance(text, str):
            return self._empty_features()
        
        # Basic features
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Word analysis
        words = text.split()
        unique_words = set(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Readability features
        syllables = self._count_syllables(text)
        flesch_score = self._calculate_flesch_score(word_count, sentence_count, syllables)
        
        # Domain-specific features
        space_score = sum(1 for word in words if word.lower() in self.space_keywords)
        science_score = sum(1 for word in words if word.lower() in self.science_keywords)
        
        # Structural features
        paragraph_count = len(text.split('\n\n'))
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Complexity features
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'unique_word_ratio': len(unique_words) / word_count if word_count > 0 else 0,
            'avg_word_length': avg_word_length,
            'syllable_count': syllables,
            'flesch_score': flesch_score,
            'space_score': space_score,
            'science_score': science_score,
            'paragraph_count': paragraph_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'long_word_ratio': long_word_ratio
        }
    
    def _empty_features(self):
        """Return empty feature set"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'unique_word_ratio': 0, 'avg_word_length': 0, 'syllable_count': 0,
            'flesch_score': 0, 'space_score': 0, 'science_score': 0,
            'paragraph_count': 0, 'exclamation_count': 0, 'question_count': 0,
            'long_word_ratio': 0
        }
    
    def _count_syllables(self, text):
        """Count syllables in text"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(count, 1)
    
    def _calculate_flesch_score(self, words, sentences, syllables):
        """Calculate Flesch reading ease score"""
        if words == 0 or sentences == 0:
            return 0
        return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    
    def create_features(self, train_texts, test_texts):
        """Create comprehensive feature matrix"""
        print("🔧 Creating features...")
        
        # TF-IDF features
        print("  • Creating TF-IDF features...")
        all_texts = []
        for text_pair in train_texts:
            all_texts.extend(text_pair)
        for text_pair in test_texts:
            all_texts.extend(text_pair)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Handcrafted features
        print("  • Creating handcrafted features...")
        handcrafted_features = []
        for text_pair in train_texts + test_texts:
            for text in text_pair:
                features = self.extract_text_features(text)
                handcrafted_features.append(list(features.values()))
        
        handcrafted_array = np.array(handcrafted_features)
        
        # Combine features
        print("  • Combining features...")
        combined_features = np.hstack([
            tfidf_features.toarray(),
            handcrafted_array
        ])
        
        print(f"✅ Created feature matrix: {combined_features.shape}")
        return combined_features
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        print("🚀 Training models...")
        
        # Split features for training
        n_train = len(y_train)
        X_train_combined = X_train[:n_train*2]  # Both texts for each training sample
        
        # Create labels for both texts
        y_train_expanded = []
        for label in y_train:
            if label == 1:
                y_train_expanded.extend([1, 0])  # text_1 is real, text_2 is fake
            else:
                y_train_expanded.extend([0, 1])  # text_2 is real, text_1 is fake
        
        # Train models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            print(f"  • Training {name}...")
            model.fit(X_train_combined, y_train_expanded)
            self.models[name] = model
        
        print("✅ Models trained successfully!")
    
    def predict(self, X_test, test_ids):
        """Generate predictions for test data"""
        print("🎯 Generating predictions...")
        
        predictions = []
        
        for i, test_id in enumerate(test_ids):
            # Get features for both texts
            text1_idx = i * 2
            text2_idx = i * 2 + 1
            
            # Get predictions from each model
            model_predictions = []
            for model_name, model in self.models.items():
                pred1 = model.predict_proba(X_test[text1_idx:text1_idx+1])[0][1]
                pred2 = model.predict_proba(X_test[text2_idx:text2_idx+1])[0][1]
                
                # Choose the text with higher probability of being real
                if pred1 > pred2:
                    model_predictions.append(1)
                else:
                    model_predictions.append(2)
            
            # Ensemble prediction (majority vote)
            final_prediction = max(set(model_predictions), key=model_predictions.count)
            predictions.append(final_prediction)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_ids,
            'real_text_id': predictions
        })
        
        print(f"✅ Generated predictions for {len(test_ids)} test samples")
        return submission_df
    
    def run_pipeline(self):
        """Run the complete competition pipeline"""
        print("🚀 STARTING COMPETITION PIPELINE")
        print("=" * 60)
        
        try:
            # Load data
            train_texts, train_labels = self.load_training_data()
            test_texts, test_ids = self.load_test_data()
            
            # Create features
            X_combined = self.create_features(train_texts, test_texts)
            
            # Split features for training and testing
            n_train = len(train_labels)
            X_train = X_combined[:n_train*2]
            X_test = X_combined[n_train*2:]
            
            # Train models
            self.train_models(X_train, train_labels)
            
            # Generate predictions
            submission_df = self.predict(X_test, test_ids)
            
            # Save submission
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            submission_file = f"competition_submission_{timestamp}.csv"
            submission_df.to_csv(submission_file, index=False)
            
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📁 Submission saved: {submission_file}")
            print(f"📊 Test samples: {len(test_ids)}")
            print(f"🔍 Training samples: {len(train_labels)}")
            
            return submission_df
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    runner = CompetitionPipelineRunner()
    submission = runner.run_pipeline()
