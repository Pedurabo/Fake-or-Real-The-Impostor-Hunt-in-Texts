#!/usr/bin/env python3
"""
Test Enhanced Model Training
Validate the enhanced model training functionality
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from modules.enhanced_model_trainer import EnhancedModelTrainer
from modules.enhanced_feature_selector import EnhancedFeatureSelector

def test_enhanced_model_training():
    """Test the enhanced model training"""
    print("🧪 TESTING ENHANCED MODEL TRAINING")
    print("=" * 60)
    
    # Create sample data
    print("1. Creating sample data...")
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # Generate synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some features being predictive
    y = np.random.choice([1, 2], size=n_samples)
    
    # Make some features predictive
    X['feature_0'] = y + np.random.normal(0, 0.1, n_samples)
    X['feature_1'] = y * 2 + np.random.normal(0, 0.1, n_samples)
    X['feature_2'] = (y == 1).astype(int) + np.random.normal(0, 0.1, n_samples)
    
    print(f"   ✓ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    print("\n2. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Validation set: {X_val.shape[0]} samples")
    
    # Initialize enhanced feature selector
    print("\n3. Running enhanced feature selection...")
    selector = EnhancedFeatureSelector()
    X_train_selected, selected_features = selector.maximize_performance_selection(
        X_train, y_train, target_features=20, cv_folds=3
    )
    
    # Apply same selection to validation set
    X_val_selected = X_val[selected_features]
    
    print(f"   ✓ Selected {len(selected_features)} features")
    
    # Initialize enhanced model trainer
    print("\n4. Initializing enhanced model trainer...")
    trainer = EnhancedModelTrainer()
    
    # Train enhanced models
    print("\n5. Training enhanced models...")
    try:
        best_model, best_score = trainer.train_enhanced_models(
            X_train_selected, X_val_selected, y_train, y_val, selected_features
        )
        
        print(f"\n✅ SUCCESS: Model training completed!")
        print(f"🏆 Best model: {trainer.best_model_name}")
        print(f"🚀 Best F1 Score: {trainer.best_score:.4f}")
        
        # Get training summary
        print("\n6. Getting training summary...")
        summary = trainer.get_training_summary()
        print(f"   ✓ Summary generated with {len(summary)} sections")
        
        # Display model comparison
        if 'summary_table' in summary:
            print("\n📊 MODEL PERFORMANCE COMPARISON:")
            print(summary['summary_table'].to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Model training failed - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_model_training()
    if success:
        print("\n🎉 All tests passed! Enhanced model training is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
