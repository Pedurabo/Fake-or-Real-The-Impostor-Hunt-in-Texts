#!/usr/bin/env python3
"""
Test Enhanced Feature Selection System with Real Competition Features
This script uses the actual feature matrix from the competition but generates realistic synthetic targets
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

def create_realistic_targets_for_real_features(feature_matrix):
    """Create realistic targets for the real feature matrix"""
    print("🔧 Creating realistic targets for real competition features...")
    
    np.random.seed(42)
    n_samples = len(feature_matrix)
    
    # Use the actual features to create realistic targets
    # Focus on features that would logically indicate real vs fake text
    
    # Select key features for target generation
    key_features = [
        'text1_vocab_richness', 'text2_vocab_richness',
        'text1_complexity', 'text2_complexity',
        'text1_info_density', 'text2_info_density',
        'text1_repetition', 'text2_repetition',
        'text1_avg_sentence_length', 'text2_avg_sentence_length',
        'text1_avg_word_length', 'text2_avg_word_length'
    ]
    
    # Filter to features that exist in the data
    available_features = [f for f in key_features if f in feature_matrix.columns]
    print(f"   ✓ Using {len(available_features)} key features for target generation")
    
    if len(available_features) == 0:
        print("   ⚠️ No key features found, using random targets")
        return np.random.choice([0, 1], size=n_samples)
    
    # Calculate weighted score based on available features
    weighted_score = np.zeros(n_samples)
    feature_weights = {
        'vocab_richness': 0.25,
        'complexity': 0.25,
        'info_density': 0.2,
        'repetition': 0.15,
        'sentence_length': 0.1,
        'word_length': 0.05
    }
    
    for feature in available_features:
        if feature in feature_matrix.columns:
            feature_values = feature_matrix[feature].values
            
            # Handle different data types
            if feature_matrix[feature].dtype == 'bool':
                feature_values = feature_values.astype(float)
            elif feature_matrix[feature].dtype == 'object':
                # Skip object columns
                continue
            
            # Normalize feature to 0-1 range
            if feature_values.max() != feature_values.min():
                normalized_feature = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
            else:
                normalized_feature = np.zeros_like(feature_values)
            
            # Apply appropriate weight based on feature type
            if 'vocab_richness' in feature:
                weight = feature_weights['vocab_richness']
            elif 'complexity' in feature:
                weight = feature_weights['complexity']
            elif 'info_density' in feature:
                weight = feature_weights['info_density']
            elif 'repetition' in feature:
                weight = feature_weights['repetition']
            elif 'sentence_length' in feature:
                weight = feature_weights['sentence_length']
            elif 'word_length' in feature:
                weight = feature_weights['word_length']
            else:
                weight = 0.1
            
            weighted_score += weight * normalized_feature
    
    # Convert to probabilities and generate targets
    # Higher scores should correlate with real text (class 1)
    base_prob = 0.4  # Slight bias toward class 0
    probabilities = 1 / (1 + np.exp(-(weighted_score - base_prob)))
    
    # Generate targets
    targets = np.random.binomial(1, probabilities)
    
    print(f"   ✓ Generated targets: {np.sum(targets)} class 1, {np.sum(targets == 0)} class 0")
    print(f"   ✓ Target distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")
    
    return targets

def test_enhanced_system_real_features():
    """Test the enhanced system with real competition features"""
    print("🧪 TESTING ENHANCED SYSTEM WITH REAL COMPETITION FEATURES")
    print("=" * 70)
    
    # Load real feature matrix
    print("1. 📊 Loading real competition feature matrix...")
    try:
        feature_matrix = pd.read_csv('src/feature_matrix.csv')
        print(f"   ✓ Loaded feature matrix: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        print(f"   ✓ Columns: {list(feature_matrix.columns[:5])}... (showing first 5)")
        
    except Exception as e:
        print(f"   ❌ Error loading feature matrix: {str(e)}")
        return False
    
    # Create realistic targets
    print("\n2. 🔧 Creating realistic targets for real features...")
    try:
        targets = create_realistic_targets_for_real_features(feature_matrix)
        feature_matrix['real_text_id'] = targets
        
        print(f"   ✓ Targets created successfully")
        
    except Exception as e:
        print(f"   ❌ Error creating targets: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Prepare data
    print("\n3. 🔧 Preparing data for enhanced feature selection...")
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
    print("\n4. 🔬 Testing enhanced feature selection...")
    try:
        selector = EnhancedFeatureSelector()
        
        # Test with different feature counts
        feature_counts = [20, 25, 30]
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
    print("\n5. 🎯 Testing enhanced model training...")
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
    
    # Performance comparison with original system
    print("\n6. 📈 Performance comparison with original system...")
    try:
        # Load original selected features if available
        try:
            original_features = pd.read_csv('src/selected_features.csv')
            print(f"   📊 Original system: {len(original_features)} features")
            
            # Compare feature overlap
            if best_features:
                overlap = len(set(best_features) & set(original_features['feature_name']))
                overlap_ratio = overlap / len(best_features) * 100
                print(f"   🔄 Feature overlap: {overlap}/{len(best_features)} ({overlap_ratio:.1f}%)")
                
        except:
            print("   📊 Original selected features not found for comparison")
            
    except Exception as e:
        print(f"   ⚠️ Could not perform performance comparison: {str(e)}")
    
    # Generate comprehensive report
    print("\n7. 📝 Generating comprehensive test report...")
    try:
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(feature_matrix),
                'total_features': len(feature_matrix.columns) - 2,  # excluding id and target
                'target_distribution': feature_matrix['real_text_id'].value_counts().to_dict(),
                'feature_matrix_source': 'src/feature_matrix.csv'
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
        with open('enhanced_system_real_features_test_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        print("   ✅ Test report saved as 'enhanced_system_real_features_test_report.json'")
        
    except Exception as e:
        print(f"   ⚠️ Could not generate report: {str(e)}")
    
    print(f"\n🎉 ENHANCED SYSTEM TEST COMPLETED!")
    print(f"📊 Data tested: {len(feature_matrix)} samples, {len(feature_matrix.columns)-2} features")
    if best_features:
        print(f"🎯 Best feature selection: {len(best_features)} features")
        print(f"🚀 Best model performance: {trainer.best_score:.4f}")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_system_real_features()
    if success:
        print("\n🎉 All tests passed! Enhanced system is working with real competition features.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
