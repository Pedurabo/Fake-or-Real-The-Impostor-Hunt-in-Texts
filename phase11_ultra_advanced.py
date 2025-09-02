#!/usr/bin/env python3
"""
Phase 11 Ultra-Advanced Competition Submission Generator
Key improvements: Feature selection, advanced ensembles, better regularization
Target: Push score from Phase 10 toward 0.70+ competition score
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
import re

class Phase11UltraAdvancedSubmissionGenerator:
    """Phase 11 ultra-advanced submission generator with key improvements"""
    
    def __init__(self, data_path="src/temp_data/data"):
        self.data_path = data_path
        self.test_path = os.path.join(data_path, "test")
        self.train_path = os.path.join(data_path, "train")
        
        # Initialize models and features
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.ensemble_model = None
        self.feature_selector = None
        
    def extract_advanced_text_features(self, text):
        """Extract advanced text features with improved analysis"""
        if not text or not isinstance(text, str):
            return self._empty_features()
        
        # Basic features
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Advanced word analysis
        words = text.split()
        unique_words = set(words)
        
        # Enhanced readability scores
        syllables = sum(self._count_syllables(word) for word in words)
        flesch_score = self._calculate_flesch_score(word_count, sentence_count, syllables)
        
        # Vocabulary complexity
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        long_word_ratio = sum(1 for word in words if len(word) > 6) / len(words) if words else 0
        
        # Advanced structural features
        paragraph_count = text.count('\n\n') + 1
        quote_count = text.count('"') // 2 + text.count("'") // 2
        bullet_point_count = sum(1 for line in text.split('\n') if line.strip().startswith(('•', '-', '*', '1.', '2.')))
        
        # Technical indicators
        technical_terms = [word for word in words if len(word) >= 8 and word.isalpha()]
        acronyms = [word for word in words if word.isupper() and 2 <= len(word) <= 5]
        
        # Citation patterns
        citation_patterns = [r'\[\d+\]', r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4}\)']
        citation_count = sum(len(re.findall(pattern, text)) for pattern in citation_patterns)
        
        # Semantic features
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0
        formality_score = self._calculate_formality_score(words)
        
        return {
            'char_count': char_count, 'word_count': word_count, 'sentence_count': sentence_count,
            'syllable_count': syllables, 'avg_word_length': avg_word_length, 'long_word_ratio': long_word_ratio,
            'unique_word_ratio': unique_word_ratio, 'flesch_score': flesch_score,
            'paragraph_count': paragraph_count, 'quote_count': quote_count, 'bullet_point_count': bullet_point_count,
            'technical_term_count': len(technical_terms), 'acronym_count': len(acronyms),
            'citation_count': citation_count, 'formality_score': formality_score
        }
    
    def _empty_features(self):
        """Return empty feature dictionary"""
        return {k: 0 for k in ['char_count', 'word_count', 'sentence_count', 'syllable_count', 
                               'avg_word_length', 'long_word_ratio', 'unique_word_ratio', 'flesch_score',
                               'paragraph_count', 'quote_count', 'bullet_point_count', 'technical_term_count',
                               'acronym_count', 'citation_count', 'formality_score']}
    
    def _count_syllables(self, word):
        """Improved syllable counting"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count
    
    def _calculate_flesch_score(self, word_count, sentence_count, syllable_count):
        """Calculate Flesch Reading Ease score"""
        if sentence_count == 0 or word_count == 0:
            return 0
        return 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
    
    def _calculate_formality_score(self, words):
        """Calculate formality score"""
        formal_words = ['therefore', 'consequently', 'furthermore', 'moreover', 'thus', 'hence']
        informal_words = ['so', 'but', 'and', 'also', 'then', 'now', 'well', 'okay']
        
        formal_count = sum(1 for word in words if word.lower() in formal_words)
        informal_count = sum(1 for word in words if word.lower() in informal_words)
        
        total = formal_count + informal_count
        if total == 0:
            return 0.5
        return formal_count / total
    
    def create_enhanced_tfidf_features(self, texts, is_training=True):
        """Create enhanced TF-IDF features"""
        print("📝 Creating enhanced TF-IDF features...")
        
        if is_training:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1500,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.9,
                stop_words='english',
                sublinear_tf=True
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        print(f"✅ Created TF-IDF features with shape: {tfidf_features.shape}")
        return tfidf_features
    
    def create_count_vectorizer_features(self, texts, is_training=True):
        """Create CountVectorizer features"""
        print("🔢 Creating CountVectorizer features...")
        
        if is_training:
            self.count_vectorizer = CountVectorizer(
                max_features=800,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9,
                stop_words='english'
            )
            count_features = self.count_vectorizer.fit_transform(texts)
        else:
            count_features = self.count_vectorizer.transform(texts)
        
        print(f"✅ Created CountVectorizer features with shape: {count_features.shape}")
        return count_features
    
    def create_handcrafted_features(self, texts):
        """Create advanced handcrafted features"""
        print("🔧 Creating advanced handcrafted features...")
        
        feature_list = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"   Processing article {i+1}/{len(texts)}")
            
            features = self.extract_advanced_text_features(text)
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list)
        feature_df = feature_df.fillna(0)
        
        print(f"✅ Created {feature_df.shape[1]} advanced handcrafted features")
        return feature_df
    
    def apply_feature_selection(self, X_train, y_train, X_test):
        """Apply feature selection to reduce overfitting"""
        print("🎯 Applying feature selection...")
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Statistical feature selection
        k_best = SelectKBest(score_func=f_classif, k=min(800, X_train.shape[1]))
        X_train_selected = k_best.fit_transform(X_train, y_train_encoded)
        X_test_selected = k_best.transform(X_test)
        
        # Model-based feature selection
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=min(600, X_train_selected.shape[1])
        )
        X_train_final = selector.fit_transform(X_train_selected, y_train_encoded)
        X_test_final = selector.transform(X_test_selected)
        
        self.feature_selector = (k_best, selector)
        
        print(f"✅ Feature selection applied: {X_train.shape[1]} → {X_train_final.shape[1]} features")
        return X_train_final, X_test_final
    
    def combine_features(self, tfidf_features, count_features, handcrafted_features):
        """Combine all feature types"""
        print("🔗 Combining features...")
        
        tfidf_dense = tfidf_features.toarray()
        count_dense = count_features.toarray()
        
        combined_features = np.hstack([tfidf_dense, count_dense, handcrafted_features])
        
        print(f"✅ Combined features shape: {combined_features.shape}")
        return combined_features
    
    def train_advanced_models(self, X_train, y_train):
        """Train advanced models with optimized parameters"""
        print("🤖 Training advanced models...")
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Define optimized models
        base_models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=400, learning_rate=0.03, max_depth=6,
                min_samples_split=10, min_samples_leaf=4, subsample=0.7, random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=250, max_depth=10, min_samples_split=8,
                min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1
            )),
            ('lr', LogisticRegression(
                C=0.5, max_iter=2000, random_state=42, solver='liblinear'
            ))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create stacking classifier
        self.ensemble_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.ensemble_model, X_train, y_train_encoded, cv=skf, scoring='f1')
        
        print(f"✅ Stacking Ensemble CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.ensemble_model.fit(X_train, y_train_encoded)
        
        return {'stacking_ensemble': cv_scores.mean()}
    
    def generate_predictions(self, X_test):
        """Generate predictions using ensemble model"""
        print("🎯 Generating predictions...")
        
        predictions = self.ensemble_model.predict(X_test)
        predictions_original = self.label_encoder.inverse_transform(predictions)
        
        print(f"✅ Generated {len(predictions)} predictions")
        return predictions_original
    
    def create_submission_file(self, test_ids, predictions, filename=None):
        """Create submission file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase11_ultra_advanced_submission_{timestamp}.csv"
        
        submission_df = pd.DataFrame({
            'id': test_ids,
            'real_text_id': predictions
        })
        
        submission_df = submission_df.sort_values('id')
        submission_df.to_csv(filename, index=False)
        print(f"✅ Submission file saved: {filename}")
        
        return filename
    
    def run(self):
        """Run the complete Phase 11 generation process"""
        print("🚀 Starting Phase 11 Ultra-Advanced Submission Generation...")
        print("=" * 70)
        
        try:
            # Load data
            test_articles, test_ids = self.load_test_data()
            train_articles, train_labels = self.load_training_data()
            
            if not test_articles or not train_articles:
                print("❌ No data found.")
                return
            
            # Feature engineering
            print("\n🔧 Feature Engineering Phase...")
            train_tfidf = self.create_enhanced_tfidf_features(train_articles, is_training=True)
            test_tfidf = self.create_enhanced_tfidf_features(test_articles, is_training=False)
            
            train_count = self.create_count_vectorizer_features(train_articles, is_training=True)
            test_count = self.create_count_vectorizer_features(test_articles, is_training=False)
            
            train_handcrafted = self.create_handcrafted_features(train_articles)
            test_handcrafted = self.create_handcrafted_features(test_articles)
            
            # Combine features
            X_train = self.combine_features(train_tfidf, train_count, train_handcrafted)
            X_test = self.combine_features(test_tfidf, test_count, test_handcrafted)
            
            # Feature selection
            X_train_selected, X_test_selected = self.apply_feature_selection(X_train, train_labels, X_test)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)
            
            # Train models
            print("\n🤖 Model Training Phase...")
            cv_scores = self.train_advanced_models(X_train_scaled, train_labels)
            
            # Generate predictions
            print("\n🎯 Prediction Generation...")
            predictions = self.generate_predictions(X_test_scaled)
            
            # Create submission
            print("\n📁 Submission Creation...")
            submission_file = self.create_submission_file(test_ids, predictions)
            
            print("\n" + "=" * 70)
            print("🎉 Phase 11 Complete!")
            print(f"📁 Submission: {submission_file}")
            print(f"🎯 CV F1: {cv_scores['stacking_ensemble']:.4f}")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_test_data(self):
        """Load test data"""
        print("🔍 Loading test data...")
        
        test_articles = []
        test_ids = []
        
        for item in os.listdir(self.test_path):
            if os.path.isdir(os.path.join(self.test_path, item)):
                article_path = os.path.join(self.test_path, item)
                
                for file_item in os.listdir(article_path):
                    if file_item.endswith('.txt'):
                        with open(os.path.join(article_path, file_item), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        test_articles.append(text)
                        test_ids.append(int(item.split('_')[1]))
                        break
        
        print(f"✅ Loaded {len(test_articles)} test articles")
        return test_articles, test_ids
    
    def load_training_data(self):
        """Load training data"""
        print("🔍 Loading training data...")
        
        train_csv_path = os.path.join(self.data_path, "train.csv")
        if not os.path.exists(train_csv_path):
            print(f"❌ Training CSV not found")
            return [], []
        
        train_df = pd.read_csv(train_csv_path)
        print(f"✅ Loaded {len(train_df)} training labels")
        
        train_articles = []
        train_labels = []
        
        for _, row in train_df.iterrows():
            article_id = row['id']
            label = row['real_text_id']
            
            article_dir = f"article_{article_id:04d}"
            article_path = os.path.join(self.train_path, article_dir)
            
            if os.path.exists(article_path):
                for file_item in os.listdir(article_path):
                    if file_item.endswith('.txt'):
                        with open(os.path.join(article_path, file_item), 'r', encoding='utf-8') as f:
                            text_content = f.read().strip()
                        
                        train_articles.append(text_content)
                        train_labels.append(label)
                        break
        
        print(f"✅ Loaded {len(train_articles)} training articles")
        return train_articles, train_labels

if __name__ == "__main__":
    generator = Phase11UltraAdvancedSubmissionGenerator()
    generator.run()
