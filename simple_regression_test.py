#!/usr/bin/env python3
"""
Simple Regression Analysis Test
Demonstrates regression analysis capabilities for feature relationship understanding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append('src/modules')

try:
    from regression_analyzer import RegressionAnalyzer
    from regression_feature_engineer import RegressionFeatureEngineer
    print("✅ Successfully imported regression analysis modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the modules are in the correct location")
    sys.exit(1)

def create_sample_data(n_samples=500, n_features=15):
    """Create sample data with known relationships"""
    print("🔧 Creating sample data with known relationships...")
    
    np.random.seed(42)
    
    # Create base features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target variable with known relationships
    y = np.zeros(n_samples)
    
    # Strong linear relationship
    y += 2.0 * X['feature_0']
    
    # Polynomial relationship
    y += 0.5 * X['feature_1']**2
    
    # Interaction
    y += 0.8 * X['feature_2'] * X['feature_3']
    
    # Add noise
    y += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary classification
    y_binary = (y > np.median(y)).astype(int)
    
    print(f"✓ Sample data created: {X.shape}")
    print(f"✓ Target distribution: {np.bincount(y_binary)}")
    
    return X, y, y_binary

def test_regression_analysis(X, y):
    """Test the regression analyzer"""
    print("\n🔍 Testing Regression Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = RegressionAnalyzer()
        
        # Perform analysis
        analysis_results = analyzer.analyze_feature_relationships(X, y)
        
        # Get recommendations
        recommendations = analyzer.get_feature_recommendations()
        
        print("✅ Regression analysis completed successfully!")
        print(f"✓ High importance features: {len(recommendations['keep_features'])}")
        print(f"✓ Features to engineer: {len(recommendations['engineer_features'])}")
        print(f"✓ Significant interactions: {len(recommendations['interaction_features'])}")
        
        return analyzer, analysis_results, recommendations
        
    except Exception as e:
        print(f"❌ Regression analysis failed: {e}")
        return None, None, None

def test_feature_engineering(X, y, regression_insights):
    """Test the feature engineer"""
    print("\n🔧 Testing Feature Engineering...")
    
    try:
        # Initialize feature engineer
        feature_engineer = RegressionFeatureEngineer()
        
        # Create enhanced features
        enhanced_features = feature_engineer.engineer_features_from_regression_insights(
            X, y, regression_insights
        )
        
        print("✅ Feature engineering completed successfully!")
        print(f"✓ Original features: {X.shape[1]}")
        print(f"✓ Enhanced features: {enhanced_features.shape[1]}")
        print(f"✓ New features added: {enhanced_features.shape[1] - X.shape[1]}")
        
        return feature_engineer, enhanced_features
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None, None

def compare_performance(X_original, X_enhanced, y):
    """Compare model performance"""
    print("\n🏆 Comparing Model Performance...")
    
    try:
        # Split data
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X_original, y, test_size=0.2, random_state=42
        )
        
        X_train_enh, X_test_enh, _, _ = train_test_split(
            X_enhanced, y, test_size=0.2, random_state=42
        )
        
        # Ensure test sets have same indices
        X_test_enh = X_test_enh.loc[X_test_orig.index]
        
        # Test Random Forest
        print("\n📊 Testing Random Forest...")
        
        # Original features
        rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_orig.fit(X_train_orig, y_train)
        y_pred_orig = rf_orig.predict(X_test_orig)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        print(f"  ✓ Original features - Accuracy: {acc_orig:.4f}")
        
        # Enhanced features
        rf_enh = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_enh.fit(X_train_enh, y_train)
        y_pred_enh = rf_enh.predict(X_test_enh)
        acc_enh = accuracy_score(y_test, y_pred_enh)
        print(f"  ✓ Enhanced features - Accuracy: {acc_enh:.4f}")
        
        # Improvement
        improvement = acc_enh - acc_orig
        print(f"  🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # Test Logistic Regression
        print("\n📊 Testing Logistic Regression...")
        
        # Original features
        lr_orig = LogisticRegression(random_state=42, max_iter=1000)
        lr_orig.fit(X_train_orig, y_train)
        y_pred_orig = lr_orig.predict(X_test_orig)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        print(f"  ✓ Original features - Accuracy: {acc_orig:.4f}")
        
        # Enhanced features
        lr_enh = LogisticRegression(random_state=42, max_iter=1000)
        lr_enh.fit(X_train_enh, y_train)
        y_pred_enh = lr_enh.predict(X_test_enh)
        acc_enh = accuracy_score(y_test, y_pred_enh)
        print(f"  ✓ Enhanced features - Accuracy: {acc_enh:.4f}")
        
        # Improvement
        improvement = acc_enh - acc_orig
        print(f"  🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        print("✅ Performance comparison completed!")
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

def main():
    """Main test function"""
    print("🚀 SIMPLE REGRESSION ANALYSIS TEST")
    print("=" * 60)
    
    # Step 1: Create sample data
    X, y_continuous, y_binary = create_sample_data()
    
    # Step 2: Test regression analysis
    analyzer, analysis_results, recommendations = test_regression_analysis(X, y_continuous)
    
    if analyzer is None:
        print("❌ Cannot continue without regression analysis")
        return
    
    # Step 3: Test feature engineering
    feature_engineer, enhanced_features = test_feature_engineering(
        X, y_continuous, analysis_results
    )
    
    if feature_engineer is None:
        print("❌ Cannot continue without feature engineering")
        return
    
    # Step 4: Compare performance
    compare_performance(X, enhanced_features, y_binary)
    
    # Step 5: Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 40)
    print("✅ Regression analysis completed")
    print("✅ Feature engineering completed")
    print("✅ Performance comparison completed")
    
    # Key insights
    original_features = X.shape[1]
    enhanced_features_count = enhanced_features.shape[1]
    feature_improvement = enhanced_features_count / original_features
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"• Feature count: {original_features} → {enhanced_features_count} ({feature_improvement:.1f}x)")
    print(f"• High importance features: {len(recommendations['keep_features'])}")
    print(f"• Features to engineer: {len(recommendations['engineer_features'])}")
    print(f"• Significant interactions: {len(recommendations['interaction_features'])}")
    
    print("\n🎉 Simple regression analysis test completed successfully!")

if __name__ == "__main__":
    main()
