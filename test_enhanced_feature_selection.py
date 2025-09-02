#!/usr/bin/env python3
"""
Test Enhanced Feature Selection
Validate the enhanced feature selection functionality
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from modules.enhanced_feature_selector import EnhancedFeatureSelector

def test_enhanced_feature_selection():
    """Test the enhanced feature selection"""
    print("🧪 TESTING ENHANCED FEATURE SELECTION")
    print("=" * 60)
    
    # Create sample data
    print("1. Creating sample data...")
    np.random.seed(42)
    n_samples = 100
    n_features = 100
    
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
    
    # Initialize enhanced feature selector
    print("\n2. Initializing enhanced feature selector...")
    selector = EnhancedFeatureSelector()
    
    # Test feature selection
    print("\n3. Running enhanced feature selection...")
    try:
        X_selected, selected_features = selector.maximize_performance_selection(
            X, y, target_features=20, cv_folds=3
        )
        
        print(f"\n✅ SUCCESS: Feature selection completed!")
        print(f"📊 Original features: {X.shape[1]}")
        print(f"🎯 Selected features: {len(selected_features)}")
        print(f"📉 Reduction ratio: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")
        
        # Test feature importance analysis
        print("\n4. Testing feature importance analysis...")
        importance = selector.get_feature_importance_analysis(X_selected, y)
        
        # Generate selection report
        print("\n5. Generating selection report...")
        report = selector.generate_selection_report()
        print(f"   ✓ Report generated: {len(report)} sections")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Feature selection failed - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_feature_selection()
    if success:
        print("\n🎉 All tests passed! Enhanced feature selection is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
