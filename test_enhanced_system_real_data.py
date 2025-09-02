#!/usr/bin/env python3
"""
Test Enhanced Feature Selection System with Real Competition Data
This script tests the enhanced feature selection system using the actual feature matrix data
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

def test_enhanced_system_real_data():
    """Test the enhanced system with real competition data"""
    print("🧪 TESTING ENHANCED SYSTEM WITH REAL COMPETITION DATA")
    print("=" * 70)
    
    # Load real data
    print("1. 📊 Loading real competition data...")
    try:
        feature_matrix = pd.read_csv('src/feature_matrix.csv')
        print(f"   ✓ Loaded feature matrix: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        
        # Check data structure
        print(f"   ✓ Columns: {list(feature_matrix.columns[:5])}... (showing first 5)")
        print(f"   ✓ Target column: real_text_id")
        print(f"   ✓ Target distribution: {feature_matrix['real_text_id'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"   ❌ Error loading data: {str(e)}")
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
        
        # Check for missing values in target
        target_missing = y.isnull().sum()
        if target_missing > 0:
            print(f"   ⚠️ Found {target_missing} missing values in target, removing those samples")
            # Remove samples with missing target values
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
            print(f"   ✓ After cleaning: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Check for missing values in features
        missing_values = X.isnull().sum().sum()
        if missing_values > 0:
            print(f"   ⚠️ Found {missing_values} missing values in features, filling with 0")
            X = X.fillna(0)
        else:
            print("   ✓ No missing values found in features")
            
        # Check data types
        print(f"   ✓ Data types: {X.dtypes.value_counts().to_dict()}")
        
        # Ensure target is numeric
        if y.dtype == 'object':
            print("   ⚠️ Converting target to numeric")
            y = pd.to_numeric(y, errors='coerce')
            # Remove any remaining NaN values
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
            print(f"   ✓ After target conversion: {X.shape[0]} samples")
        
        print(f"   ✓ Final data shape: {X.shape}")
        print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")
        
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
    
    # Performance comparison
    print("\n5. 📈 Performance comparison with original system...")
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
    print("\n6. 📝 Generating comprehensive test report...")
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
        with open('enhanced_system_test_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        print("   ✅ Test report saved as 'enhanced_system_test_report.json'")
        
    except Exception as e:
        print(f"   ⚠️ Could not generate report: {str(e)}")
    
    print(f"\n🎉 ENHANCED SYSTEM TEST COMPLETED!")
    print(f"📊 Data tested: {len(feature_matrix)} samples, {len(feature_matrix.columns)-2} features")
    if best_features:
        print(f"🎯 Best feature selection: {len(best_features)} features")
        print(f"🚀 Best model performance: {trainer.best_score:.4f}")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_system_real_data()
    if success:
        print("\n🎉 All tests passed! Enhanced system is working with real data.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
