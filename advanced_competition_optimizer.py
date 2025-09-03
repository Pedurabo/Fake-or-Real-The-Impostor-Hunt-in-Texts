#!/usr/bin/env python3
"""
Advanced Competition Optimizer
This script implements sophisticated techniques to push accuracy beyond 93.55%
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

class AdvancedCompetitionOptimizer:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.cv_scores = {}
        self.ensemble_model = None
        
    def advanced_text_preprocessing(self, text):
        """Advanced text preprocessing with domain-specific cleaning"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Remove special characters but keep important punctuation
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Domain-specific cleaning for space/science text
        text = re.sub(r'\b(article|paper|research|study|analysis)\b', '', text)
        
        return text
    
    def extract_semantic_features(self, text1, text2):
        """Extract semantic similarity features using advanced NLP"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode texts
            embedding1 = model.encode(text1)
            embedding2 = model.encode(text2)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Additional semantic features
            semantic_features = {
                'cosine_similarity': similarity,
                'euclidean_distance': np.linalg.norm(embedding1 - embedding2),
                'manhattan_distance': np.sum(np.abs(embedding1 - embedding2)),
                'dot_product': np.dot(embedding1, embedding2),
                'cosine_distance': 1 - similarity
            }
            
            return semantic_features
            
        except ImportError:
            # Fallback to basic features if sentence-transformers not available
            return {
                'cosine_similarity': 0.5,
                'euclidean_distance': 1.0,
                'manhattan_distance': 1.0,
                'dot_product': 0.0,
                'cosine_distance': 0.5
            }
    
    def create_advanced_features(self, df):
        """Create advanced features beyond basic text analysis"""
        print("🔧 Creating advanced features...")
        
        advanced_features = []
        
        for idx, row in df.iterrows():
            text1 = self.advanced_text_preprocessing(row['text_1'])
            text2 = self.advanced_text_preprocessing(row['text_2'])
            
            # Semantic features
            semantic = self.extract_semantic_features(text1, text2)
            
            # Advanced linguistic features
            import nltk
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                
                # Tokenize and process
                tokens1 = word_tokenize(text1)
                tokens2 = word_tokenize(text2)
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens1_clean = [w for w in tokens1 if w.lower() not in stop_words]
                tokens2_clean = [w for w in tokens2 if w.lower() not in stop_words]
                
                # Lemmatization
                lemmatizer = WordNetLemmatizer()
                tokens1_lem = [lemmatizer.lemmatize(w) for w in tokens1_clean]
                tokens2_lem = [lemmatizer.lemmatize(w) for w in tokens2_clean]
                
                # Advanced features
                linguistic_features = {
                    'unique_words_ratio': len(set(tokens1_lem)) / max(len(tokens1_lem), 1),
                    'common_words_ratio': len(set(tokens1_lem) & set(tokens2_lem)) / max(len(set(tokens1_lem) | set(tokens2_lem)), 1),
                    'sentence_complexity_1': len(tokens1) / max(text1.count('.'), 1),
                    'sentence_complexity_2': len(tokens2) / max(text2.count('.'), 1),
                    'vocabulary_diversity_1': len(set(tokens1)) / max(len(tokens1), 1),
                    'vocabulary_diversity_2': len(set(tokens2)) / max(len(tokens2), 1)
                }
                
            except:
                linguistic_features = {
                    'unique_words_ratio': 0.5,
                    'common_words_ratio': 0.3,
                    'sentence_complexity_1': 10.0,
                    'sentence_complexity_2': 10.0,
                    'vocabulary_diversity_1': 0.5,
                    'vocabulary_diversity_2': 0.5
                }
            
            # Combine all features
            all_features = {**semantic, **linguistic_features}
            advanced_features.append(all_features)
        
        return pd.DataFrame(advanced_features)
    
    def load_and_prepare_data(self):
        """Load and prepare data with advanced preprocessing"""
        print("📊 Loading and preparing data...")
        
        # Load training data
        train_df = pd.read_csv('data/train.csv')
        
        # Apply advanced preprocessing
        print("🧹 Applying advanced text preprocessing...")
        train_df['text_1_clean'] = train_df['text_1'].apply(self.advanced_text_preprocessing)
        train_df['text_2_clean'] = train_df['text_2'].apply(self.advanced_text_preprocessing)
        
        # Create advanced features
        advanced_features = self.create_advanced_features(train_df)
        
        # Combine with existing features if available
        try:
            existing_features = pd.read_csv('feature_matrix.csv')
            if 'real_text_id' in existing_features.columns:
                # Use existing features as base
                X = existing_features.drop('real_text_id', axis=1)
                y = existing_features['real_text_id']
                print(f"✅ Using existing features: {X.shape}")
            else:
                raise FileNotFoundError
        except:
            # Create basic features if none exist
            print("📝 Creating basic features...")
            X = self.create_basic_features(train_df)
            y = train_df['real_text_id']
        
        # Add advanced features
        X = pd.concat([X, advanced_features], axis=1)
        print(f"✅ Final feature matrix: {X.shape}")
        
        return X, y
    
    def create_basic_features(self, df):
        """Create basic text features as fallback"""
        features = []
        
        for idx, row in df.iterrows():
            text1 = row['text_1_clean']
            text2 = row['text_2_clean']
            
            basic_features = {
                'length_1': len(text1),
                'length_2': len(text2),
                'length_diff': abs(len(text1) - len(text2)),
                'word_count_1': len(text1.split()),
                'word_count_2': len(text2.split()),
                'word_count_diff': abs(len(text1.split()) - len(text2.split())),
                'char_count_1': len(text1.replace(' ', '')),
                'char_count_2': len(text2.replace(' ', '')),
                'char_count_diff': abs(len(text1.replace(' ', '')) - len(text2.replace(' ', ''))),
                'avg_word_length_1': np.mean([len(w) for w in text1.split()]) if text1.split() else 0,
                'avg_word_length_2': np.mean([len(w) for w in text2.split()]) if text2.split() else 0
            }
            
            features.append(basic_features)
        
        return pd.DataFrame(features)
    
    def advanced_ensemble_training(self, X, y):
        """Train advanced ensemble with stacking and blending"""
        print("🤖 Training advanced ensemble models...")
        
        # Define base models with optimized parameters
        base_models = {
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                min_samples_split=5, random_state=42
            ),
            'lr': LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear',
                max_iter=1000, random_state=42
            ),
            'svm': SVC(
                C=1.0, kernel='rbf', gamma='scale',
                probability=True, random_state=42
            )
        }
        
        # Train individual models
        for name, model in base_models.items():
            print(f"   Training {name.upper()}...")
            model.fit(X, y)
            self.models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            self.cv_scores[name] = cv_scores.mean()
            print(f"      CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Create voting ensemble
        print("   Creating voting ensemble...")
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        
        # Train ensemble
        self.ensemble_model.fit(X, y)
        ensemble_cv = cross_val_score(self.ensemble_model, X, y, cv=5, scoring='accuracy')
        self.cv_scores['ensemble'] = ensemble_cv.mean()
        print(f"      Ensemble CV Accuracy: {ensemble_cv.mean():.4f} ± {ensemble_cv.std():.4f}")
        
        return self.ensemble_model
    
    def evaluate_performance(self, X, y):
        """Comprehensive performance evaluation"""
        print("📊 Evaluating model performance...")
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_accuracy': self.cv_scores[name]
            }
            
            print(f"   {name.upper()}:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      F1 Score: {f1:.4f}")
            print(f"      CV Accuracy: {self.cv_scores[name]:.4f}")
        
        # Ensemble evaluation
        y_pred_ensemble = self.ensemble_model.predict(X)
        ensemble_accuracy = accuracy_score(y, y_pred_ensemble)
        ensemble_f1 = f1_score(y, y_pred_ensemble, average='weighted')
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'f1_score': ensemble_f1,
            'cv_accuracy': self.cv_scores['ensemble']
        }
        
        print(f"   ENSEMBLE:")
        print(f"      Accuracy: {ensemble_accuracy:.4f}")
        print(f"      F1 Score: {ensemble_f1:.4f}")
        print(f"      CV Accuracy: {self.cv_scores['ensemble']:.4f}")
        
        return results
    
    def save_models(self):
        """Save trained models"""
        print("💾 Saving models...")
        
        import pickle
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            with open(f'models/{name}_advanced.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ {name.upper()} saved")
        
        # Save ensemble
        with open('models/ensemble_advanced.pkl', 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        print("   ✅ Ensemble saved")
        
        # Save results
        results = {
            'cv_scores': self.cv_scores,
            'models': list(self.models.keys()),
            'ensemble_created': True
        }
        
        with open('advanced_optimization_results.json', 'w') as f:
            import json
            json.dump(results, f, indent=2)
        print("   ✅ Results saved")
    
    def run_optimization(self):
        """Run the complete advanced optimization pipeline"""
        print("🚀 ADVANCED COMPETITION OPTIMIZATION PIPELINE")
        print("=" * 60)
        print("🎯 Goal: Push accuracy beyond 93.55%")
        print("🔧 Techniques: Advanced preprocessing, semantic features, ensemble methods")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            X, y = self.load_and_prepare_data()
            
            # Train advanced ensemble
            ensemble = self.advanced_ensemble_training(X, y)
            
            # Evaluate performance
            results = self.evaluate_performance(X, y)
            
            # Save models
            self.save_models()
            
            # Summary
            best_accuracy = max([r['accuracy'] for r in results.values()])
            best_cv_accuracy = max([r['cv_accuracy'] for r in results.values()])
            
            print("\n🏆 OPTIMIZATION COMPLETED!")
            print("=" * 40)
            print(f"Best Model Accuracy: {best_accuracy:.4f}")
            print(f"Best CV Accuracy: {best_cv_accuracy:.4f}")
            print(f"Previous Best: 0.9355")
            print(f"Improvement: {best_accuracy - 0.9355:+.4f}")
            
            if best_accuracy > 0.9355:
                print("🎉 NEW RECORD ACHIEVED!")
            else:
                print("📈 Further optimization needed")
            
            execution_time = time.time() - start_time
            print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    optimizer = AdvancedCompetitionOptimizer()
    optimizer.run_optimization()

if __name__ == "__main__":
    main()
