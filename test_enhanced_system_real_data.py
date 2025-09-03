#!/usr/bin/env python3
"""
Enhanced System Test with Real Data
Testing kernel methods, dimensionality reduction, and advanced feature selection
using real training and testing data
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modules.kernel_dimensionality_reducer import KernelDimensionalityReducer
from modules.enhanced_feature_selector import EnhancedFeatureSelector
from modules.enhanced_model_trainer import EnhancedModelTrainer
from modules.data_loader import DataLoader

def load_real_data():
    """Load real training and testing data"""
    print("📊 LOADING REAL TRAINING AND TESTING DATA")
    print("=" * 60)
    
    data_paths = {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'feature_matrix': 'src/feature_matrix.csv',
        'selected_features': 'src/selected_features.csv'
    }
    
    data = {}
    
    # Try to load different data formats
    for data_type, path in data_paths.items():
        try:
            if os.path.exists(path):
                print(f"🔧 Loading {data_type} data from {path}...")
                if path.endswith('.csv'):
                    data[data_type] = pd.read_csv(path)
                    print(f"   ✓ {data_type.capitalize()} data loaded: {data[data_type].shape[0]} samples, {data[data_type].shape[1]} features")
                else:
                    print(f"   ⚠️ Unsupported file format: {path}")
            else:
                print(f"   ⚠️ File not found: {path}")
        except Exception as e:
            print(f"   ✗ Error loading {data_type} data: {str(e)}")
    
    # If no data files found, create sample data structure
    if not data:
        print("\n🔧 No real data files found. Creating sample data structure...")
        print("   Please place your training and testing data in the 'data/' directory")
        print("   Expected files: train.csv, test.csv")
        print("   Or place feature matrix in: src/feature_matrix.csv")
        
        # Create minimal sample data for demonstration
        np.random.seed(42)
        n_samples, n_features = 500, 50
        
        # Generate realistic features
        X_sample = np.random.randn(n_samples, n_features)
        
        # Add some realistic patterns
        X_sample[:, 0] = X_sample[:, 1] ** 2 + np.random.randn(n_samples) * 0.1
        X_sample[:, 2] = np.sin(X_sample[:, 3]) + np.random.randn(n_samples) * 0.1
        X_sample[:, 4] = X_sample[:, 5] * X_sample[:, 6] + np.random.randn(n_samples) * 0.1
        
        # Create realistic target variable
        y_sample = ((X_sample[:, 0] > 0) & (X_sample[:, 2] > 0) | 
                   (X_sample[:, 4] > 0)).astype(int)
        
        # Convert to DataFrame with realistic feature names
        feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        X_df = pd.DataFrame(X_sample, columns=feature_names)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_sample, test_size=0.2, random_state=42, stratify=y_sample
        )
        
        data['train'] = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
        data['test'] = X_test.copy()
        data['feature_matrix'] = X_df.copy()
        
        print(f"   ✓ Sample data created: {X_df.shape[0]} samples, {X_df.shape[1]} features")
        print(f"   ✓ Training set: {X_train.shape[0]} samples")
        print(f"   ✓ Testing set: {X_test.shape[0]} samples")
    
    return data

def prepare_real_data(data):
    """Prepare real data for analysis"""
    print("\n🔧 PREPARING REAL DATA FOR ANALYSIS")
    print("=" * 60)
    
    if 'train' in data and 'test' in data:
        # We have separate train/test files
        print("📊 Using separate train/test files...")
        
        train_data = data['train']
        test_data = data['test']
        
        # Identify target column
        target_col = None
        for col in train_data.columns:
            if col.lower() in ['target', 'label', 'class', 'y', 'real_text_id', 'is_fake']:
                target_col = col
                break
        
        if target_col is None:
            # Assume last column is target
            target_col = train_data.columns[-1]
            print(f"   ⚠️ Target column not identified, using: {target_col}")
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_test = test_data
        
        print(f"   ✓ Training features: {X_train.shape[1]}")
        print(f"   ✓ Training samples: {X_train.shape[0]}")
        print(f"   ✓ Testing samples: {X_test.shape[0]}")
        print(f"   ✓ Target distribution: {y_train.value_counts().to_dict()}")
        
    elif 'feature_matrix' in data:
        # We have a feature matrix
        print("📊 Using feature matrix...")
        
        feature_matrix = data['feature_matrix']
        
        # Check if we have selected_features with targets
        if 'selected_features' in data and 'real_text_id' in data['selected_features'].columns:
            print("   📊 Using selected_features with targets...")
            selected_data = data['selected_features']
            
            # Check for NaN values in target
            if selected_data['real_text_id'].isna().any():
                print("   ⚠️ Target column contains NaN values, cleaning...")
                # Remove rows with NaN targets
                selected_data = selected_data.dropna(subset=['real_text_id'])
                print(f"   ✓ Cleaned data: {selected_data.shape[0]} samples")
            
            # Separate features and target
            X = selected_data.drop(columns=['real_text_id'])
            y = selected_data['real_text_id']
            
            # Ensure target is numeric
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().any():
                print("   ❌ Target column contains non-numeric values")
                return None, None, None
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   ✓ Features: {X.shape[1]}")
            print(f"   ✓ Training samples: {X_train.shape[0]}")
            print(f"   ✓ Testing samples: {X_test.shape[0]}")
            print(f"   ✓ Target distribution: {y_train.value_counts().to_dict()}")
            
        else:
            print("   ⚠️ No target column found in feature matrix")
            print("   Creating synthetic targets for demonstration...")
            
            # Create synthetic targets for demonstration
            n_samples = feature_matrix.shape[0]
            np.random.seed(42)
            y = np.random.choice([1, 2], size=n_samples, p=[0.6, 0.4])
            
            # Use all features
            X = feature_matrix
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   ✓ Features: {X.shape[1]}")
            print(f"   ✓ Training samples: {X_train.shape[0]}")
            print(f"   ✓ Testing samples: {X_test.shape[0]}")
            print(f"   ✓ Target distribution: {np.bincount(y_train)}")
        
        else:
        print("   ❌ No usable data found")
        return None, None, None
    
    # Check for missing values
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    
    if missing_train > 0:
        print(f"   ⚠️ Found {missing_train} missing values in training data, filling with 0")
        X_train = X_train.fillna(0)
    
    if missing_test > 0:
        print(f"   ⚠️ Found {missing_test} missing values in testing data, filling with 0")
        X_test = X_test.fillna(0)
        
        # Ensure target is numeric
    if y_train.dtype == 'object':
            print("   ⚠️ Converting target to numeric")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        print(f"   ✓ Target encoded: {le.classes_}")
    
    return X_train, X_test, y_train

def test_kernel_methods_with_real_data(X_train, X_test, y_train):
    """Test kernel methods and dimensionality reduction with real data"""
    print("\n🚀 TESTING KERNEL METHODS WITH REAL DATA")
    print("=" * 80)
    
    # Initialize the kernel dimensionality reducer
    kernel_reducer = KernelDimensionalityReducer()
    
    print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"📊 Target distribution: {np.bincount(y_train)}")
    
    # Test 1: Kernel Feature Mapping
    print("\n" + "="*60)
    print("🔍 TEST 1: KERNEL FEATURE MAPPING")
    print("="*60)
    
    # Test different kernel methods
    kernel_methods = ['rbf', 'poly', 'sigmoid', 'cosine']
    kernel_results = {}
    
    for kernel in kernel_methods:
        try:
            print(f"\n🔧 Testing {kernel.upper()} kernel...")
            
            # Apply kernel transformation
            X_train_kernel = kernel_reducer.kernel_feature_mapping(
                X_train, y_train, method=kernel, n_components=min(200, X_train.shape[0])
            )
            
            # Apply same transformation to test data
            X_test_kernel = kernel_reducer.kernel_feature_mapping(
                X_test, None, method=kernel, n_components=min(200, X_test.shape[0])
            )
            
            kernel_results[kernel] = {
                'X_train_kernel': X_train_kernel,
                'X_test_kernel': X_test_kernel,
                'expansion_ratio': X_train_kernel.shape[1] / X_train.shape[1]
            }
            
            print(f"   ✓ {kernel.upper()} kernel applied successfully")
            print(f"   📈 Training dimensions: {X_train.shape[1]} → {X_train_kernel.shape[1]}")
            print(f"   📈 Testing dimensions: {X_test.shape[1]} → {X_test_kernel.shape[1]}")
            print(f"   📊 Expansion ratio: {X_train_kernel.shape[1] / X_train.shape[1]:.2f}x")
            
    except Exception as e:
            print(f"   ✗ {kernel.upper()} kernel failed: {str(e)}")
            kernel_results[kernel] = None
    
    # Test 2: Advanced Dimensionality Reduction
    print("\n" + "="*60)
    print("🔍 TEST 2: ADVANCED DIMENSIONALITY REDUCTION")
    print("=" * 60)
    
    # Use the best kernel method for further testing
    best_kernel = 'rbf'  # Default to RBF
    if kernel_results[best_kernel] is not None:
        print(f"\n🔧 Applying {best_kernel.upper()} kernel transformation...")
        X_train_kernel = kernel_results[best_kernel]['X_train_kernel']
        X_test_kernel = kernel_results[best_kernel]['X_test_kernel']
        
        # Test different dimensionality reduction methods
        reduction_methods = ['pca', 'svd', 'ica', 'auto']
        reduction_results = {}
        
        for method in reduction_methods:
            try:
                print(f"\n🔧 Testing {method.upper()} reduction...")
                
                # Apply dimensionality reduction to training data
                X_train_reduced = kernel_reducer.advanced_dimensionality_reduction(
                    X_train_kernel, y_train, method=method, target_dim=50
                )
                
                # Apply same reduction to test data
                X_test_reduced = kernel_reducer.advanced_dimensionality_reduction(
                    X_test_kernel, None, method=method, target_dim=50
                )
                
                reduction_results[method] = {
                    'X_train_reduced': X_train_reduced,
                    'X_test_reduced': X_test_reduced,
                    'reduction_ratio': (X_train_kernel.shape[1] - X_train_reduced.shape[1]) / X_train_kernel.shape[1]
                }
                
                print(f"   ✓ {method.upper()} reduction completed")
                print(f"   📉 Training dimensions: {X_train_kernel.shape[1]} → {X_train_reduced.shape[1]}")
                print(f"   📉 Testing dimensions: {X_test_kernel.shape[1]} → {X_test_reduced.shape[1]}")
                print(f"   📊 Reduction ratio: {((X_train_kernel.shape[1] - X_train_reduced.shape[1]) / X_train_kernel.shape[1] * 100):.1f}%")
                
            except Exception as e:
                print(f"   ✗ {method.upper()} reduction failed: {str(e)}")
                reduction_results[method] = None
        else:
        print("   ⚠️ No valid kernel results available for dimensionality reduction")
        reduction_results = {}
    
    return kernel_results, reduction_results

def evaluate_real_data_performance(X_train, X_test, y_train, kernel_results, reduction_results):
    """Evaluate performance on real data with different feature representations"""
    print("\n🎯 EVALUATING PERFORMANCE ON REAL DATA")
    print("=" * 80)
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    performance_results = {}
    
    # 1. Original features
    print(f"\n📊 Testing original features...")
    print(f"   Training shape: {X_train.shape}")
    print(f"   Testing shape: {X_test.shape}")
    
    original_scores = {}
    for model_name, model in models.items():
        try:
            # Cross-validation on training data
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            
            # Train on full training data and predict on test
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            original_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_predictions': y_pred,
                'test_probabilities': y_pred_proba
            }
            
            print(f"      {model_name}: CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
    except Exception as e:
            print(f"      {model_name}: Failed - {str(e)}")
            original_scores[model_name] = None
    
    performance_results['original'] = original_scores
    
    # 2. Kernel features
    if kernel_results:
        print(f"\n📊 Testing kernel features...")
        kernel_scores = {}
        
        for kernel_name, kernel_data in kernel_results.items():
            if kernel_data is None:
                continue
                
            print(f"      🔧 {kernel_name.upper()} kernel:")
            X_train_kernel = kernel_data['X_train_kernel']
            X_test_kernel = kernel_data['X_test_kernel']
            
            kernel_method_scores = {}
            for model_name, model in models.items():
                try:
                    # Cross-validation on kernel features
                    cv_scores = cross_val_score(model, X_train_kernel, y_train, cv=cv, scoring='f1_weighted')
                    
                    # Train on full kernel training data and predict on kernel test data
                    model.fit(X_train_kernel, y_train)
                    y_pred = model.predict(X_test_kernel)
                    y_pred_proba = model.predict_proba(X_test_kernel)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    kernel_method_scores[model_name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_predictions': y_pred,
                        'test_probabilities': y_pred_proba
                    }
                    
                    print(f"         {model_name}: CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
    except Exception as e:
                    print(f"         {model_name}: Failed - {str(e)}")
                    kernel_method_scores[model_name] = None
            
            kernel_scores[kernel_name] = kernel_method_scores
        
        performance_results['kernel'] = kernel_scores
    
    # 3. Reduced features
    if reduction_results:
        print(f"\n📊 Testing reduced features...")
        reduced_scores = {}
        
        for method_name, method_data in reduction_results.items():
            if method_data is None:
                continue
                
            print(f"      🔧 {method_name.upper()} reduction:")
            X_train_reduced = method_data['X_train_reduced']
            X_test_reduced = method_data['X_test_reduced']
            
            method_scores = {}
            for model_name, model in models.items():
                try:
                    # Cross-validation on reduced features
                    cv_scores = cross_val_score(model, X_train_reduced, y_train, cv=cv, scoring='f1_weighted')
                    
                    # Train on full reduced training data and predict on reduced test data
                    model.fit(X_train_reduced, y_train)
                    y_pred = model.predict(X_test_reduced)
                    y_pred_proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    method_scores[model_name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_predictions': y_pred,
                        'test_probabilities': y_pred_proba
                    }
                    
                    print(f"         {model_name}: CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
    except Exception as e:
                    print(f"         {model_name}: Failed - {str(e)}")
                    method_scores[model_name] = None
            
            reduced_scores[method_name] = method_scores
        
        performance_results['reduced'] = reduced_scores
    
    return performance_results

def generate_real_data_report(X_train, X_test, y_train, kernel_results, reduction_results, performance_results):
    """Generate comprehensive report for real data analysis"""
    print("\n📊 GENERATING REAL DATA ANALYSIS REPORT")
    print("=" * 80)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))
    fig.suptitle('Real Data: Kernel Methods & Dimensionality Reduction Analysis', fontsize=16)
    
    # 1. Data characteristics
    ax1 = axes[0, 0]
    data_info = [
        f'Training: {X_train.shape[0]} samples\n{X_train.shape[1]} features',
        f'Testing: {X_test.shape[0]} samples\n{X_test.shape[1]} features'
    ]
    ax1.bar(['Training', 'Testing'], [X_train.shape[0], X_test.shape[0]], 
             color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_title('Dataset Characteristics')
    ax1.set_ylabel('Number of Samples')
    
    # Add feature count as text
    for i, (train_feat, test_feat) in enumerate([(X_train.shape[1], X_test.shape[1])]):
        ax1.text(i, max(X_train.shape[0], X_test.shape[0]) * 0.1, 
                f'{train_feat} features', ha='center', va='bottom', fontweight='bold')
    
    # 2. Target distribution
    ax2 = axes[0, 1]
    target_counts = np.bincount(y_train)
    target_labels = [f'Class {i}' for i in range(len(target_counts))]
    ax2.pie(target_counts, labels=target_labels, autopct='%1.1f%%', 
             colors=['lightblue', 'lightcoral'], startangle=90)
    ax2.set_title('Target Distribution')
    
    # 3. Kernel expansion ratios
    ax3 = axes[0, 2]
    if kernel_results:
        kernel_names = list(kernel_results.keys())
        expansion_ratios = []
        for kernel in kernel_names:
            if kernel_results[kernel] is not None:
                expansion_ratios.append(kernel_results[kernel]['expansion_ratio'])
            else:
                expansion_ratios.append(0)
        
        ax3.bar(kernel_names, expansion_ratios, color='lightgreen', alpha=0.7)
        ax3.set_title('Kernel Expansion Ratios')
        ax3.set_ylabel('Expansion Ratio')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Dimensionality reduction ratios
    ax4 = axes[1, 0]
    if reduction_results:
        method_names = list(reduction_results.keys())
        reduction_ratios = []
        for method in method_names:
            if reduction_results[method] is not None:
                reduction_ratios.append(reduction_results[method]['reduction_ratio'])
            else:
                reduction_ratios.append(0)
        
        ax4.bar(method_names, reduction_ratios, color='lightcoral', alpha=0.7)
        ax4.set_title('Dimensionality Reduction Ratios')
        ax4.set_ylabel('Reduction Ratio')
        ax4.tick_params(axis='x', rotation=45)
    
    # 5. Performance comparison (Random Forest)
    ax5 = axes[1, 1]
    model_name = 'Random Forest'
    
    # Original performance
    original_score = 0
    if 'original' in performance_results and model_name in performance_results['original']:
        original_score = performance_results['original'][model_name]['cv_mean']
    
    # Best kernel performance
    best_kernel_score = 0
    if 'kernel' in performance_results:
        for kernel_name, kernel_data in performance_results['kernel'].items():
            if model_name in kernel_data and kernel_data[model_name] is not None:
                best_kernel_score = max(best_kernel_score, kernel_data[model_name]['cv_mean'])
    
    # Best reduced performance
    best_reduced_score = 0
    if 'reduced' in performance_results:
        for method_name, method_data in performance_results['reduced'].items():
            if model_name in method_data and method_data[model_name] is not None:
                best_reduced_score = max(best_reduced_score, method_data[model_name]['cv_mean'])
    
    approaches = ['Original', 'Best Kernel', 'Best Reduced']
    scores = [original_score, best_kernel_score, best_reduced_score]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    ax5.bar(approaches, scores, color=colors, alpha=0.7)
    ax5.set_title(f'Performance Comparison ({model_name})')
    ax5.set_ylabel('CV F1 Score')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Feature correlation heatmap (if reduced dimensions are reasonable)
    ax6 = axes[1, 2]
    if reduction_results and 'auto' in reduction_results and reduction_results['auto'] is not None:
        X_reduced = reduction_results['auto']['X_train_reduced']
        if X_reduced.shape[1] <= 20:
            corr_matrix = np.corrcoef(X_reduced.T)
            im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            ax6.set_title('Reduced Features Correlation Matrix')
            ax6.set_xlabel('Feature Index')
            ax6.set_ylabel('Feature Index')
            plt.colorbar(im, ax=ax6)
        else:
            ax6.text(0.5, 0.5, 'Too many features\nfor correlation plot', 
                     ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Feature Correlation Matrix')
    else:
        ax6.text(0.5, 0.5, 'No reduced features\navailable', 
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    # Save the comprehensive report
    save_path = 'real_data_kernel_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   📁 Real data analysis saved to: {save_path}")
    
    plt.show()
    
    return fig

def main():
    """Main test function with real data"""
    print("🚀 ENHANCED SYSTEM TEST WITH REAL DATA")
    print("=" * 80)
    
    try:
        # Step 1: Load real data
        data = load_real_data()
        
        # Step 2: Prepare real data
        X_train, X_test, y_train = prepare_real_data(data)
        
        if X_train is None:
            print("\n❌ Could not prepare data for analysis")
            return
        
        # Step 3: Test kernel methods with real data
        kernel_results, reduction_results = test_kernel_methods_with_real_data(X_train, X_test, y_train)
        
        # Step 4: Evaluate performance on real data
        performance_results = evaluate_real_data_performance(X_train, X_test, y_train, kernel_results, reduction_results)
        
        # Step 5: Generate comprehensive report
        fig = generate_real_data_report(X_train, X_test, y_train, kernel_results, reduction_results, performance_results)
        
        # Final Summary
        print("\n" + "="*80)
        print("🎯 FINAL SUMMARY - REAL DATA ANALYSIS")
        print("=" * 80)
        
        print(f"\n📊 Data Summary:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Training features: {X_train.shape[1]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        print(f"   Target distribution: {np.bincount(y_train)}")
        
        print(f"\n📊 Kernel Methods Results:")
        if kernel_results:
            for kernel_name, kernel_data in kernel_results.items():
                if kernel_data is not None:
                    print(f"   {kernel_name.upper()}: {kernel_data['expansion_ratio']:.2f}x expansion")
        
        print(f"\n📊 Dimensionality Reduction Results:")
        if reduction_results:
            for method_name, method_data in reduction_results.items():
                if method_data is not None:
                    print(f"   {method_name.upper()}: {method_data['reduction_ratio']:.1%} reduction")
        
        print(f"\n📊 Performance Summary (Random Forest):")
        if 'original' in performance_results and 'Random Forest' in performance_results['original']:
            original_score = performance_results['original']['Random Forest']['cv_mean']
            print(f"   Original features: {original_score:.4f}")
        
        if 'kernel' in performance_results:
            best_kernel_score = 0
            best_kernel_name = 'None'
            for kernel_name, kernel_data in performance_results['kernel'].items():
                if 'Random Forest' in kernel_data and kernel_data['Random Forest'] is not None:
                    score = kernel_data['Random Forest']['cv_mean']
                    if score > best_kernel_score:
                        best_kernel_score = score
                        best_kernel_name = kernel_name
            if best_kernel_score > 0:
                print(f"   Best kernel ({best_kernel_name}): {best_kernel_score:.4f}")
        
        if 'reduced' in performance_results:
            best_reduced_score = 0
            best_reduced_method = 'None'
            for method_name, method_data in performance_results['reduced'].items():
                if 'Random Forest' in method_data and method_data['Random Forest'] is not None:
                    score = method_data['Random Forest']['cv_mean']
                    if score > best_reduced_score:
                        best_reduced_score = score
                        best_reduced_method = method_name
            if best_reduced_score > 0:
                print(f"   Best reduced ({best_reduced_method}): {best_reduced_score:.4f}")
        
        print("\n✅ Real data analysis completed successfully!")
        print("📁 Check 'real_data_kernel_analysis.png' for visualizations")
        
    except Exception as e:
        print(f"\n❌ Real data analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
