#!/usr/bin/env python3
"""
Enhanced Integration Script
Simple integration of enhanced feature selection and model training
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from modules.enhanced_feature_selector import EnhancedFeatureSelector
from modules.enhanced_model_trainer import EnhancedModelTrainer

def run_enhanced_pipeline():
    """Run the enhanced pipeline"""
    print("🚀 ENHANCED INTEGRATION PIPELINE")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample data...")
    np.random.seed(42)
    n_samples = 200
    n_features = 80
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y = np.random.choice([1, 2], size=n_samples)
    
    # Make some features predictive
    X['feature_0'] = y + np.random.normal(0, 0.1, n_samples)
    X['feature_1'] = y * 2 + np.random.normal(0, 0.1, n_samples)
    X['feature_2'] = (y == 1).astype(int) + np.random.normal(0, 0.1, n_samples)
    
    print(f"   ✓ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    print("\n2. Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Enhanced feature selection
    print("\n3. Enhanced feature selection...")
    selector = EnhancedFeatureSelector()
    X_train_selected, selected_features = selector.maximize_performance_selection(
        X_train, y_train, target_features=25, cv_folds=3
    )
    
    X_val_selected = X_val[selected_features]
    
    # Enhanced model training
    print("\n4. Enhanced model training...")
    trainer = EnhancedModelTrainer()
    best_model, best_score = trainer.train_enhanced_models(
        X_train_selected, X_val_selected, y_train, y_val, selected_features
    )
    
    print(f"\n✅ ENHANCED PIPELINE COMPLETED!")
    print(f"🏆 Best model: {trainer.best_model_name}")
    print(f"🚀 Best F1 Score: {best_score:.4f}")
    print(f"📊 Selected features: {len(selected_features)}")
    
    return best_model, selected_features, best_score

if __name__ == "__main__":
    best_model, selected_features, best_score = run_enhanced_pipeline()
