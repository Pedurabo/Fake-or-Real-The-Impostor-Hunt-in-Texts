#!/usr/bin/env python3
"""
Test Optimized Feature Selection
Verify that the performance optimizations work correctly
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modules.advanced_feature_selector import AdvancedFeatureSelector
from modules.enhanced_feature_selector import EnhancedFeatureSelector

def create_test_data(n_samples=1000, n_features=200):
    """Create test data for performance testing"""
    print(f"🔧 Creating test data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
    # Generate realistic features
    X = np.random.randn(n_samples, n_features)
    
    # Add some realistic patterns
    X[:, 0] = X[:, 1] ** 2 + np.random.randn(n_samples) * 0.1
    X[:, 2] = np.sin(X[:, 3]) + np.random.randn(n_samples) * 0.1
    X[:, 4] = X[:, 5] * X[:, 6] + np.random.randn(n_samples) * 0.1
    
    # Create realistic target variable
    y = ((X[:, 0] > 0) & (X[:, 2] > 0) | (X[:, 4] > 0)).astype(int)
    
    # Convert to DataFrame with realistic feature names
    feature_names = [f'feature_{i:03d}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y

def test_advanced_feature_selector():
    """Test the optimized advanced feature selector"""
    print("\n" + "="*60)
    print("🧪 TESTING OPTIMIZED ADVANCED FEATURE SELECTOR")
    print("="*60)
    
    # Create test data
    X, y = create_test_data(n_samples=800, n_features=150)
    
    # Initialize selector
    selector = AdvancedFeatureSelector()
    
    # Test comprehensive selection with timing
    print(f"\n📊 Testing with {X.shape[0]} samples, {X.shape[1]} features")
    
    start_time = time.time()
    try:
        X_selected, selected_features = selector.comprehensive_feature_selection(X, y, target_features=50)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Advanced feature selection completed in {elapsed_time:.2f} seconds")
        print(f"📊 Selected {len(selected_features)} features")
        print(f"📉 Reduction ratio: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")
        
        return True, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Advanced feature selection failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        return False, elapsed_time

def test_enhanced_feature_selector():
    """Test the optimized enhanced feature selector"""
    print("\n" + "="*60)
    print("🧪 TESTING OPTIMIZED ENHANCED FEATURE SELECTOR")
    print("="*60)
    
    # Create test data
    X, y = create_test_data(n_samples=800, n_features=150)
    
    # Initialize selector
    selector = EnhancedFeatureSelector()
    
    # Test performance selection with timing
    print(f"\n📊 Testing with {X.shape[0]} samples, {X.shape[1]} features")
    
    start_time = time.time()
    try:
        X_selected, selected_features = selector.maximize_performance_selection(X, y, target_features=50, cv_folds=3)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Enhanced feature selection completed in {elapsed_time:.2f} seconds")
        print(f"📊 Selected {len(selected_features)} features")
        print(f"📉 Reduction ratio: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")
        
        return True, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Enhanced feature selection failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        return False, elapsed_time

def test_large_dataset():
    """Test with a larger dataset to trigger optimization paths"""
    print("\n" + "="*60)
    print("🧪 TESTING LARGE DATASET OPTIMIZATION")
    print("="*60)
    
    # Create larger test data
    X, y = create_test_data(n_samples=2500, n_features=400)
    
    # Initialize selector
    selector = AdvancedFeatureSelector()
    
    # Test comprehensive selection with timing
    print(f"\n📊 Testing with {X.shape[0]} samples, {X.shape[1]} features")
    print("⚠️ This should trigger the fast selection path...")
    
    start_time = time.time()
    try:
        X_selected, selected_features = selector.comprehensive_feature_selection(X, y, target_features=100)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Large dataset feature selection completed in {elapsed_time:.2f} seconds")
        print(f"📊 Selected {len(selected_features)} features")
        print(f"📉 Reduction ratio: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")
        
        return True, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Large dataset feature selection failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        return False, elapsed_time

def main():
    """Main test function"""
    print("🚀 TESTING OPTIMIZED FEATURE SELECTION PERFORMANCE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Advanced Feature Selector
    success, time_taken = test_advanced_feature_selector()
    results['advanced'] = {'success': success, 'time': time_taken}
    
    # Test 2: Enhanced Feature Selector
    success, time_taken = test_enhanced_feature_selector()
    results['enhanced'] = {'success': success, 'time': time_taken}
    
    # Test 3: Large Dataset Optimization
    success, time_taken = test_large_dataset()
    results['large_dataset'] = {'success': success, 'time': time_taken}
    
    # Summary
    print("\n" + "="*80)
    print("📊 PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{test_name:20s}: {status} in {result['time']:.2f} seconds")
    
    # Check if all tests passed
    all_passed = all(result['success'] for result in results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Feature selection optimization successful!")
        print("🚀 Sequential feature selection should now run much faster!")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    main()
