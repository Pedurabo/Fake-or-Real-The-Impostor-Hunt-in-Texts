#!/usr/bin/env python3
"""
Quick Kernel Methods Demo
This script demonstrates kernel methods and dimensionality reduction without time-consuming optimization
"""

import sys
import os
sys.path.append('src')

def main():
    print("🚀 QUICK KERNEL METHODS DEMO")
    print("=" * 50)
    
    try:
        from modules.kernel_dimensionality_reducer import KernelDimensionalityReducer
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import time
        
        # Load real data
        print("📊 Loading real data...")
        try:
            # Try to load selected_features.csv first
            if os.path.exists('src/selected_features.csv'):
                data = pd.read_csv('src/selected_features.csv')
                print(f"✅ Loaded selected_features.csv with {data.shape[0]} rows and {data.shape[1]} columns")
                
                # Check if we have target labels
                if 'real_text_id' in data.columns and data['real_text_id'].notna().any():
                    # Use real targets
                    X = data.drop(['real_text_id'], axis=1)
                    y = data['real_text_id'].astype(int)
                    print(f"✅ Using real targets: {len(y.unique())} classes")
                else:
                    # Generate synthetic targets
                    X = data
                    y = np.random.randint(0, 2, size=len(data))
                    print("⚠️ No valid targets found, using synthetic binary targets")
            else:
                # Fallback to feature_matrix.csv
                data = pd.read_csv('src/feature_matrix.csv')
                print(f"✅ Loaded feature_matrix.csv with {data.shape[0]} rows and {data.shape[1]} columns")
                X = data
                y = np.random.randint(0, 2, size=len(data))
                print("⚠️ Using synthetic binary targets")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Generating synthetic data for demonstration...")
            # Generate synthetic data
            np.random.seed(42)
            X = np.random.randn(100, 20)
            y = np.random.randint(0, 2, 100)
            print(f"✅ Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Clean data
        X = X.fillna(0)
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number])
        
        print(f"📊 Final data shape: {X.shape}")
        print(f"🎯 Target distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"📈 Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Initialize kernel reducer
        print("\n🚀 Initializing Kernel Dimensionality Reducer...")
        kernel_reducer = KernelDimensionalityReducer()
        
        # Test kernel feature mapping
        print("\n🎯 STAGE 1: KERNEL FEATURE MAPPING")
        print("-" * 40)
        
        start_time = time.time()
        
        # Test different kernel methods
        kernel_methods = ['rbf', 'polynomial', 'sigmoid', 'cosine', 'laplacian']
        kernel_results = {}
        
        for method in kernel_methods:
            print(f"🔧 Testing {method.upper()} kernel...")
            try:
                X_kernel = kernel_reducer.kernel_feature_mapping(
                    X_train, y_train, method=method, n_components=min(50, X_train.shape[1])
                )
                expansion_ratio = X_kernel.shape[1] / X_train.shape[1]
                kernel_results[method] = expansion_ratio
                print(f"   ✅ {method.upper()}: {X_kernel.shape[1]} features ({expansion_ratio:.2f}x expansion)")
            except Exception as e:
                print(f"   ❌ {method.upper()}: Error - {e}")
                kernel_results[method] = 0
        
        kernel_time = time.time() - start_time
        print(f"⏱️ Kernel mapping completed in {kernel_time:.2f} seconds")
        
        # Find best kernel
        best_kernel = max(kernel_results, key=kernel_results.get)
        print(f"🏆 Best kernel method: {best_kernel.upper()} ({kernel_results[best_kernel]:.2f}x expansion)")
        
        # Apply best kernel transformation
        print(f"\n🎯 STAGE 2: APPLYING BEST KERNEL ({best_kernel.upper()})")
        print("-" * 50)
        
        X_train_kernel = kernel_reducer.kernel_feature_mapping(
            X_train, y_train, method=best_kernel, n_components=min(100, X_train.shape[1] * 2)
        )
        
        # Dimensionality reduction
        print("\n🎯 STAGE 3: DIMENSIONALITY REDUCTION")
        print("-" * 40)
        
        start_time = time.time()
        
        # Test different reduction methods
        reduction_methods = ['pca', 'kernel_pca', 'truncated_svd', 'fast_ica']
        reduction_results = {}
        
        for method in reduction_methods:
            print(f"🔧 Testing {method.upper()} reduction...")
            try:
                X_reduced = kernel_reducer.advanced_dimensionality_reduction(
                    X_train_kernel, y_train, method=method, n_components=min(20, X_train_kernel.shape[1])
                )
                reduction_ratio = X_reduced.shape[1] / X_train_kernel.shape[1]
                reduction_results[method] = reduction_ratio
                print(f"   ✅ {method.upper()}: {X_reduced.shape[1]} features ({reduction_ratio:.2f}x reduction)")
            except Exception as e:
                print(f"   ❌ {method.upper()}: Error - {e}")
                reduction_results[method] = 1
        
        reduction_time = time.time() - start_time
        print(f"⏱️ Dimensionality reduction completed in {reduction_time:.2f} seconds")
        
        # Find best reduction method
        best_reduction = min(reduction_results, key=reduction_results.get)
        print(f"🏆 Best reduction method: {best_reduction.upper()} ({reduction_results[best_reduction]:.2f}x reduction)")
        
        # Apply best reduction
        X_train_final = kernel_reducer.advanced_dimensionality_reduction(
            X_train_kernel, y_train, method=best_reduction, n_components=min(20, X_train_kernel.shape[1])
        )
        
        # Quick model comparison
        print("\n🎯 STAGE 4: QUICK MODEL COMPARISON")
        print("-" * 40)
        
        # Original features
        print("🔍 Testing with original features...")
        rf_original = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_original.fit(X_train, y_train)
        y_pred_original = rf_original.predict(X_test)
        acc_original = accuracy_score(y_test, y_pred_original)
        print(f"   ✅ Original features accuracy: {acc_original:.4f}")
        
        # Kernel + reduced features
        print("🔍 Testing with kernel + reduced features...")
        # Transform test data
        X_test_kernel = kernel_reducer.kernel_feature_mapping(
            X_test, y_test, method=best_kernel, n_components=X_train_kernel.shape[1]
        )
        X_test_final = kernel_reducer.advanced_dimensionality_reduction(
            X_test_kernel, y_test, method=best_reduction, n_components=X_train_final.shape[1]
        )
        
        rf_kernel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_kernel.fit(X_train_final, y_train)
        y_pred_kernel = rf_kernel.predict(X_test_final)
        acc_kernel = accuracy_score(y_test, y_pred_kernel)
        print(f"   ✅ Kernel + reduced features accuracy: {acc_kernel:.4f}")
        
        # Performance summary
        print("\n🏆 PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"📊 Original features: {X_train.shape[1]} → Accuracy: {acc_original:.4f}")
        print(f"🚀 Kernel features: {X_train_kernel.shape[1]} → Accuracy: {acc_kernel:.4f}")
        print(f"📉 Reduced features: {X_train_final.shape[1]} → Accuracy: {acc_kernel:.4f}")
        
        improvement = ((acc_kernel - acc_original) / acc_original) * 100
        print(f"📈 Performance improvement: {improvement:+.2f}%")
        
        # Feature efficiency
        original_efficiency = acc_original / X_train.shape[1]
        kernel_efficiency = acc_kernel / X_train_final.shape[1]
        efficiency_improvement = ((kernel_efficiency - original_efficiency) / original_efficiency) * 100
        
        print(f"⚡ Feature efficiency improvement: {efficiency_improvement:+.2f}%")
        
        print("\n✅ QUICK KERNEL DEMO COMPLETED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
