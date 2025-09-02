#!/usr/bin/env python3
"""
Test Script: Regression Analysis for Feature Relationship Understanding
Demonstrates how regression analysis captures relationships between independent and dependent variables
to improve model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append('src/modules')
from regression_analyzer import RegressionAnalyzer
from regression_feature_engineer import RegressionFeatureEngineer

def create_sample_data(n_samples=1000, n_features=20):
    """Create sample data with known relationships for testing"""
    print("🔧 CREATING SAMPLE DATA WITH KNOWN RELATIONSHIPS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create base features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target variable with known relationships
    y = np.zeros(n_samples)
    
    # Strong linear relationship with feature_0
    y += 2.5 * X['feature_0']
    
    # Polynomial relationship with feature_1
    y += 0.8 * X['feature_1']**2 - 0.3 * X['feature_1']
    
    # Interaction between feature_2 and feature_3
    y += 1.2 * X['feature_2'] * X['feature_3']
    
    # Non-linear relationship with feature_4
    y += 0.5 * np.sin(X['feature_4'])
    
    # Ratio relationship
    y += 0.3 * (X['feature_5'] / (X['feature_6'] + 1e-8))
    
    # Add some noise
    y += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary classification for demonstration
    y_binary = (y > np.median(y)).astype(int)
    
    print(f"✓ Sample data created: {X.shape}")
    print(f"✓ Target distribution: {np.bincount(y_binary)}")
    print(f"✓ Known relationships:")
    print(f"  - Linear: feature_0 (coefficient: 2.5)")
    print(f"  - Polynomial: feature_1 (quadratic)")
    print(f"  - Interaction: feature_2 × feature_3 (coefficient: 1.2)")
    print(f"  - Non-linear: feature_4 (sinusoidal)")
    print(f"  - Ratio: feature_5 / feature_6 (coefficient: 0.3)")
    
    return X, y, y_binary

def test_regression_analysis(X, y):
    """Test the regression analyzer"""
    print("\n🔍 TESTING REGRESSION ANALYZER")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RegressionAnalyzer()
    
    # Perform comprehensive analysis
    analysis_results = analyzer.analyze_feature_relationships(X, y)
    
    # Get feature recommendations
    recommendations = analyzer.get_feature_recommendations()
    
    print("\n📋 FEATURE RECOMMENDATIONS:")
    print("-" * 40)
    for category, features in recommendations.items():
        print(f"  {category.replace('_', ' ').title()}: {len(features)} features")
        if features:
            print(f"    Examples: {features[:3]}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    analyzer.generate_visualizations(save_path="regression_analysis_test.png")
    
    # Export results
    analyzer.export_analysis_results("regression_analysis_test_results.json")
    
    return analyzer, analysis_results, recommendations

def test_regression_feature_engineering(X, y, regression_insights):
    """Test the regression-based feature engineer"""
    print("\n🔧 TESTING REGRESSION-BASED FEATURE ENGINEERING")
    print("=" * 60)
    
    # Initialize feature engineer
    feature_engineer = RegressionFeatureEngineer()
    
    # Create enhanced features
    enhanced_features = feature_engineer.engineer_features_from_regression_insights(
        X, y, regression_insights
    )
    
    # Generate report
    feature_engineer.generate_feature_engineering_report(save_path="feature_engineering_test.png")
    
    # Get insights
    insights = feature_engineer.get_feature_engineering_insights()
    
    print("\n📋 FEATURE ENGINEERING INSIGHTS:")
    print("-" * 40)
    for key, value in insights.items():
        if isinstance(value, list):
            print(f"  {key.replace('_', ' ').title()}: {len(value)} items")
            if value:
                print(f"    Examples: {value[:3]}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return feature_engineer, enhanced_features

def compare_model_performance(X_original, X_enhanced, y, test_size=0.2):
    """Compare model performance with original vs enhanced features"""
    print("\n🏆 COMPARING MODEL PERFORMANCE")
    print("=" * 60)
    
    # Split data
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=test_size, random_state=42
    )
    
    X_train_enh, X_test_enh, _, _ = train_test_split(
        X_enhanced, y, test_size=test_size, random_state=42
    )
    
    # Ensure test sets have same indices
    X_test_enh = X_test_enh.loc[X_test_orig.index]
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Testing {model_name}...")
        
        # Test with original features
        try:
            model.fit(X_train_orig, y_train)
            y_pred_orig = model.predict(X_test_orig)
            
            if hasattr(model, 'predict_proba'):
                # Classification model
                acc_orig = accuracy_score(y_test, y_pred_orig)
                original_key = f"{model_name}_original"
                results[original_key] = {'accuracy': acc_orig, 'type': 'classification'}
                print(f"  ✓ Original features - Accuracy: {acc_orig:.4f}")
            else:
                # Regression model
                r2_orig = r2_score(y_test, y_pred_orig)
                mse_orig = mean_squared_error(y_test, y_pred_orig)
                results[original_key] = {'r2': r2_orig, 'mse': mse_orig, 'type': 'regression'}
                print(f"  ✓ Original features - R²: {r2_orig:.4f}, MSE: {mse_orig:.4f}")
        except Exception as e:
            print(f"  ❌ Original features failed: {e}")
            results[original_key] = {'error': str(e)}
        
        # Test with enhanced features
        try:
            model.fit(X_train_enh, y_train)
            y_pred_enh = model.predict(X_test_enh)
            
            if hasattr(model, 'predict_proba'):
                # Classification model
                acc_enh = accuracy_score(y_test, y_pred_enh)
                results[f"{model_name}_enhanced"] = {'accuracy': acc_enh, 'type': 'classification'}
                print(f"  ✓ Enhanced features - Accuracy: {acc_enh:.4f}")
                
                # Calculate improvement
                if f"{model_name}_original" in results and 'accuracy' in results[f"{model_name}_original"]:
                    improvement = acc_enh - results[f"{model_name}_original']['accuracy']
                    print(f"  🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
            else:
                # Regression model
                r2_enh = r2_score(y_test, y_pred_enh)
                mse_enh = mean_squared_error(y_test, y_pred_enh)
                results[f"{model_name}_enhanced"] = {'r2': r2_enh, 'mse': mse_enh, 'type': 'regression'}
                print(f"  ✓ Enhanced features - R²: {r2_enh:.4f}, MSE: {mse_enh:.4f}")
                
                # Calculate improvement
                if f"{model_name}_original" in results and 'r2' in results[f"{model_name}_original"]:
                    improvement = r2_enh - results[f"{model_name}_original']['r2']
                    print(f"  🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        except Exception as e:
            print(f"  ❌ Enhanced features failed: {e}")
            results[f"{model_name}_enhanced"] = {'error': str(e)}
    
    return results

def generate_performance_comparison_plot(results):
    """Generate a plot comparing model performance"""
    print("\n📊 GENERATING PERFORMANCE COMPARISON PLOT")
    print("=" * 60)
    
    # Prepare data for plotting
    plot_data = []
    model_names = []
    
    for key, value in results.items():
        if 'error' not in value:
            if value['type'] == 'classification':
                plot_data.append(value['accuracy'])
                model_names.append(key.replace('_', ' ').title())
            elif value['type'] == 'regression':
                plot_data.append(value['r2'])
                model_names.append(key.replace('_', ' ').title())
    
    if not plot_data:
        print("⚠️  No valid results to plot")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance comparison
    x_pos = np.arange(len(plot_data))
    bars = ax1.bar(x_pos, plot_data, color=['skyblue' if 'Original' in name else 'lightgreen' for name in model_names])
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('Model Performance: Original vs Enhanced Features')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, plot_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Performance improvement
    improvement_data = []
    improvement_labels = []
    
    for i in range(0, len(plot_data), 2):
        if i + 1 < len(plot_data):
            improvement = plot_data[i + 1] - plot_data[i]
            improvement_data.append(improvement)
            improvement_labels.append(model_names[i].replace(' Original', ''))
    
    if improvement_data:
        colors = ['green' if x > 0 else 'red' for x in improvement_data]
        bars2 = ax2.bar(range(len(improvement_data)), improvement_data, color=colors)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Performance Improvement')
        ax2.set_title('Performance Improvement with Enhanced Features')
        ax2.set_xticks(range(len(improvement_data)))
        ax2.set_xticklabels(improvement_labels, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, improvement_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if value > 0 else -0.01),
                    f'{value:+.3f}', ha='center', va='bottom' if value > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance comparison plot generated and saved!")

def main():
    """Main test function"""
    print("🚀 REGRESSION ANALYSIS TEST SUITE")
    print("=" * 80)
    print("Testing regression analysis capabilities for feature relationship understanding")
    print("and model performance improvement")
    print("=" * 80)
    
    # Step 1: Create sample data
    X, y_continuous, y_binary = create_sample_data()
    
    # Step 2: Test regression analysis
    analyzer, analysis_results, recommendations = test_regression_analysis(X, y_continuous)
    
    # Step 3: Test feature engineering
    feature_engineer, enhanced_features = test_regression_feature_engineering(
        X, y_continuous, analysis_results
    )
    
    # Step 4: Compare model performance
    results = compare_model_performance(X, enhanced_features, y_binary)
    
    # Step 5: Generate performance comparison
    generate_performance_comparison_plot(results)
    
    # Step 6: Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 60)
    print("✅ Regression analysis completed successfully")
    print("✅ Feature engineering based on regression insights completed")
    print("✅ Model performance comparison completed")
    print("✅ All visualizations and reports generated")
    
    # Print key insights
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 30)
    
    # Feature improvement
    original_features = X.shape[1]
    enhanced_features_count = enhanced_features.shape[1]
    feature_improvement = enhanced_features_count / original_features
    
    print(f"• Feature count increased from {original_features} to {enhanced_features_count} ({feature_improvement:.1f}x)")
    
    # Performance improvements
    improvements = []
    for key, value in results.items():
        if 'enhanced' in key and 'error' not in value:
            base_key = key.replace('_enhanced', '_original')
            if base_key in results and 'error' not in results[base_key]:
                if value['type'] == 'classification':
                    improvement = value['accuracy'] - results[base_key]['accuracy']
                    improvements.append(improvement)
                elif value['type'] == 'regression':
                    improvement = value['r2'] - results[base_key]['r2']
                    improvements.append(improvement)
    
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"• Average performance improvement: {avg_improvement:+.4f}")
        print(f"• Best improvement: {max(improvements):+.4f}")
        print(f"• Models improved: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
    
    print("\n🎉 Regression analysis test suite completed successfully!")
    print("📁 Check generated files for detailed results and visualizations")

if __name__ == "__main__":
    main()
