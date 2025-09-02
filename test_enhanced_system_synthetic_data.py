#!/usr/bin/env python3
"""
Test Enhanced Feature Selection System with Synthetic Competition Data
This script creates realistic synthetic data similar to the competition and tests the enhanced system
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from modules.enhanced_feature_selector import EnhancedFeatureSelector
from modules.enhanced_model_trainer import EnhancedModelTrainer

def create_synthetic_competition_data(n_samples=1000):
    """Create realistic synthetic competition data"""
    print("🔧 Creating synthetic competition data...")
    
    np.random.seed(42)
    
    # Create realistic text analysis features similar to the competition
    data = {}
    
    # Basic text features
    data['text_1_length'] = np.random.randint(100, 3000, n_samples)
    data['text_2_length'] = np.random.randint(100, 3000, n_samples)
    data['text_1_word_count'] = np.random.randint(20, 500, n_samples)
    data['text_2_word_count'] = np.random.randint(20, 500, n_samples)
    
    # Derived features
    data['length_difference'] = data['text_1_length'] - data['text_2_length']
    data['length_ratio'] = data['text_1_length'] / (data['text_2_length'] + 1e-8)
    data['word_count_difference'] = data['text_1_word_count'] - data['text_2_word_count']
    data['word_count_ratio'] = data['text_1_word_count'] / (data['text_2_word_count'] + 1e-8)
    
    # Text quality features
    data['text_1_quality'] = np.random.choice([True, False], n_samples)
    data['text_2_quality'] = np.random.choice([True, False], n_samples)
    
    # Detailed text features
    data['text1_char_count'] = data['text_1_length'] + np.random.randint(-50, 50, n_samples)
    data['text2_char_count'] = data['text_2_length'] + np.random.randint(-50, 50, n_samples)
    
    # Punctuation and formatting
    data['text1_punctuation'] = np.random.randint(0, 50, n_samples)
    data['text2_punctuation'] = np.random.randint(0, 50, n_samples)
    data['text1_uppercase'] = np.random.randint(0, 100, n_samples)
    data['text2_uppercase'] = np.random.randint(0, 100, n_samples)
    data['text1_numbers'] = np.random.randint(0, 20, n_samples)
    data['text2_numbers'] = np.random.randint(0, 20, n_samples)
    
    # Vocabulary features
    data['text1_unique_words'] = np.random.randint(10, 300, n_samples)
    data['text2_unique_words'] = np.random.randint(10, 300, n_samples)
    data['text1_vocab_richness'] = data['text1_unique_words'] / (data['text_1_word_count'] + 1e-8)
    data['text2_vocab_richness'] = data['text2_unique_words'] / (data['text_2_word_count'] + 1e-8)
    
    # Sentence features
    data['text1_sentence_count'] = np.random.randint(1, 30, n_samples)
    data['text2_sentence_count'] = np.random.randint(1, 30, n_samples)
    data['text1_avg_sentence_length'] = data['text_1_word_count'] / (data['text1_sentence_count'] + 1e-8)
    data['text2_avg_sentence_length'] = data['text_2_word_count'] / (data['text2_sentence_count'] + 1e-8)
    
    # Word features
    data['text1_avg_word_length'] = np.random.uniform(4.0, 7.0, n_samples)
    data['text2_avg_word_length'] = np.random.uniform(4.0, 7.0, n_samples)
    
    # Space and formatting
    data['text1_space_terms'] = np.random.randint(0, 10, n_samples)
    data['text2_space_terms'] = np.random.randint(0, 10, n_samples)
    data['text1_space_ratio'] = data['text1_space_terms'] / (data['text_1_word_count'] + 1e-8)
    data['text2_space_ratio'] = data['text2_space_terms'] / (data['text_2_word_count'] + 1e-8)
    
    # Complexity features
    data['text1_complexity'] = np.random.uniform(10.0, 40.0, n_samples)
    data['text2_complexity'] = np.random.uniform(10.0, 40.0, n_samples)
    data['text1_info_density'] = np.random.uniform(0.08, 0.15, n_samples)
    data['text2_info_density'] = np.random.uniform(0.08, 0.15, n_samples)
    
    # Repetition features
    data['text1_repetition'] = np.random.uniform(0.6, 0.9, n_samples)
    data['text2_repetition'] = np.random.uniform(0.6, 0.9, n_samples)
    
    # Additional derived features
    data['vocab_difference'] = data['text1_unique_words'] - data['text2_unique_words']
    data['vocab_ratio'] = data['text1_unique_words'] / (data['text2_unique_words'] + 1e-8)
    data['space_term_difference'] = data['text1_space_terms'] - data['text2_space_terms']
    data['space_term_ratio'] = data['text1_space_terms'] / (data['text2_space_terms'] + 1e-8)
    data['complexity_difference'] = data['text1_complexity'] - data['text2_complexity']
    
    # Create target variable with some relationship to features
    # Make it realistic: some texts are more likely to be real based on features
    base_prob = 0.5
    feature_weights = {
        'text1_vocab_richness': 0.3,
        'text2_vocab_richness': 0.3,
        'text1_complexity': 0.2,
        'text2_complexity': 0.2,
        'text1_info_density': 0.15,
        'text2_info_density': 0.15
    }
    
    # Calculate weighted score
    weighted_score = np.zeros(n_samples)
    for feature, weight in feature_weights.items():
        # Normalize feature to 0-1 range
        feature_values = np.array(data[feature])
        normalized_feature = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-8)
        weighted_score += weight * normalized_feature
    
    # Convert to probabilities and generate targets
    probabilities = 1 / (1 + np.exp(-(weighted_score - base_prob)))
    data['real_text_id'] = np.random.binomial(1, probabilities)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add ID column
    df['id'] = [f'article_{i:04d}' for i in range(n_samples)]
    
    # Reorder columns to match competition format
    id_cols = ['id', 'real_text_id']
    feature_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + feature_cols]
    
    print(f"   ✓ Created synthetic dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"   ✓ Target distribution: {df['real_text_id'].value_counts().to_dict()}")
    
    return df

def test_enhanced_system_synthetic_data():
    """Test the enhanced system with synthetic competition data"""
    print("🧪 TESTING ENHANCED SYSTEM WITH SYNTHETIC COMPETITION DATA")
    print("=" * 70)
    
    # Create synthetic data
    print("1. 🔧 Creating synthetic competition data...")
    try:
        feature_matrix = create_synthetic_competition_data(n_samples=800)
        print(f"   ✓ Dataset created successfully")
        
    except Exception as e:
        print(f"   ❌ Error creating data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Prepare data
    print("\n2. 🔧 Preparing data for enhanced feature selection...")
    try:
        # Remove non-feature columns
        feature_cols = [col for col in feature_matrix.columns if col not in ['id', 'real_text_id']]
        X = feature_matrix[feature_cols]
        y = feature_matrix['real_text_id']
        
        print(f"   ✓ Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   ✓ Target vector: {len(y)} samples")
        print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")
        
        # Check for missing values
        missing_values = X.isnull().sum().sum()
        if missing_values > 0:
            print(f"   ⚠️ Found {missing_values} missing values, filling with 0")
            X = X.fillna(0)
        else:
            print("   ✓ No missing values found")
            
        # Check data types
        print(f"   ✓ Data types: {X.dtypes.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"   ❌ Error preparing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test enhanced feature selection
    print("\n3. 🔬 Testing enhanced feature selection...")
    try:
        selector = EnhancedFeatureSelector()
        
        # Test with different feature counts
        feature_counts = [15, 20, 25]
        best_features = None
        best_score = 0
        
        for target_features in feature_counts:
            print(f"\n   🎯 Testing {target_features} features...")
            
            X_selected, selected_features = selector.maximize_performance_selection(
                X, y, target_features=target_features, cv_folds=3
            )
            
            print(f"   ✓ Selected {len(selected_features)} features")
            print(f"   ✓ Feature reduction: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")
            
            # Store best result
            if len(selected_features) > 0:
                best_features = selected_features
                best_score = target_features
        
        if best_features:
            print(f"\n   🏆 Best feature selection: {len(best_features)} features")
            print(f"   📊 Selected features: {best_features[:10]}... (showing first 10)")
            
    except Exception as e:
        print(f"   ❌ Error in feature selection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test enhanced model training
    print("\n4. 🎯 Testing enhanced model training...")
    try:
        # Use the best selected features
        if best_features:
            X_selected = X[best_features]
            
            # Split data for training
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y, test_size=0.3, random_state=42, stratify=y
            )
            
            print(f"   ✓ Training set: {X_train.shape[0]} samples")
            print(f"   ✓ Validation set: {X_val.shape[0]} samples")
            
            # Initialize enhanced model trainer
            trainer = EnhancedModelTrainer()
            
            # Train enhanced models
            best_model, best_score = trainer.train_enhanced_models(
                X_train, X_val, y_train, y_val, best_features
            )
            
            print(f"\n   ✅ Enhanced model training completed!")
            print(f"   🏆 Best model: {trainer.best_model_name}")
            print(f"   🚀 Best F1 Score: {trainer.best_score:.4f}")
            
            # Get training summary
            summary = trainer.get_training_summary()
            print(f"   📊 Summary generated with {len(summary)} sections")
            
        else:
            print("   ⚠️ No features selected, skipping model training")
            
    except Exception as e:
        print(f"   ❌ Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Generate comprehensive report
    print("\n5. 📝 Generating comprehensive test report...")
    try:
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(feature_matrix),
                'total_features': len(feature_matrix.columns) - 2,  # excluding id and target
                'target_distribution': feature_matrix['real_text_id'].value_counts().to_dict()
            },
            'enhanced_system_results': {
                'best_feature_count': len(best_features) if best_features else 0,
                'selected_features': best_features if best_features else [],
                'best_model_name': trainer.best_model_name if 'trainer' in locals() else None,
                'best_f1_score': trainer.best_score if 'trainer' in locals() else None
            },
            'test_status': 'COMPLETED'
        }
        
        # Save report
        with open('enhanced_system_synthetic_test_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        print("   ✅ Test report saved as 'enhanced_system_synthetic_test_report.json'")
        
    except Exception as e:
        print(f"   ⚠️ Could not generate report: {str(e)}")
    
    print(f"\n🎉 ENHANCED SYSTEM TEST COMPLETED!")
    print(f"📊 Data tested: {len(feature_matrix)} samples, {len(feature_matrix.columns)-2} features")
    if best_features:
        print(f"🎯 Best feature selection: {len(best_features)} features")
        print(f"🚀 Best model performance: {trainer.best_score:.4f}")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_system_synthetic_data()
    if success:
        print("\n🎉 All tests passed! Enhanced system is working with synthetic data.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
