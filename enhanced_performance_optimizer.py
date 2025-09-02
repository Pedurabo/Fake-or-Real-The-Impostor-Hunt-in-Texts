#!/usr/bin/env python3
"""
Enhanced Performance Optimizer
Comprehensive performance optimization using enhanced feature selection and model training
"""

import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from modules.enhanced_feature_selector import EnhancedFeatureSelector
from modules.enhanced_model_trainer import EnhancedModelTrainer

class EnhancedPerformanceOptimizer:
    """
    Comprehensive performance optimizer for maximum competition score
    """
    
    def __init__(self):
        self.feature_selector = EnhancedFeatureSelector()
        self.model_trainer = EnhancedModelTrainer()
        self.optimization_results = {}
        self.performance_history = {}
        
    def optimize_performance(self, X, y, target_features_range=[20, 30, 40, 50], cv_folds=5):
        """
        Optimize performance using different feature counts
        """
        print("🚀 ENHANCED PERFORMANCE OPTIMIZATION")
        print("=" * 70)
        
        original_features = X.shape[1]
        print(f"📊 Original features: {original_features}")
        print(f"🎯 Testing feature counts: {target_features_range}")
        
        best_overall_score = 0
        best_config = {}
        
        for target_features in target_features_range:
            print(f"\n{'='*20} TESTING {target_features} FEATURES {'='*20}")
            
            try:
                # Step 1: Enhanced feature selection
                print(f"\n1. 🔬 Enhanced feature selection for {target_features} features...")
                X_selected, selected_features = self.feature_selector.maximize_performance_selection(
                    X, y, target_features=target_features, cv_folds=cv_folds
                )
                
                # Step 2: Split data
                print(f"\n2. 📊 Splitting data...")
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_selected, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Step 3: Enhanced model training
                print(f"\n3. 🎯 Enhanced model training...")
                best_model, best_score = self.model_trainer.train_enhanced_models(
                    X_train, X_val, y_train, y_val, selected_features
                )
                
                # Step 4: Store results
                config_results = {
                    'target_features': target_features,
                    'selected_features': selected_features,
                    'best_model_name': self.model_trainer.best_model_name,
                    'best_f1_score': self.model_trainer.best_score,
                    'feature_reduction_ratio': ((original_features - target_features) / original_features * 100),
                    'training_summary': self.model_trainer.get_training_summary()
                }
                
                self.optimization_results[target_features] = config_results
                
                # Update best overall
                if self.model_trainer.best_score > best_overall_score:
                    best_overall_score = self.model_trainer.best_score
                    best_config = config_results.copy()
                
                print(f"\n✅ Configuration {target_features} features completed!")
                print(f"   🏆 Best model: {self.model_trainer.best_model_name}")
                print(f"   🚀 Best F1 Score: {self.model_trainer.best_score:.4f}")
                
            except Exception as e:
                print(f"\n❌ ERROR in configuration {target_features}: {str(e)}")
                continue
        
        # Final results
        print(f"\n{'='*70}")
        print("🏆 FINAL OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        
        print(f"Best overall performance: {best_overall_score:.4f}")
        print(f"Best configuration: {best_config['target_features']} features")
        print(f"Best model: {best_config['best_model_name']}")
        
        return best_config, self.optimization_results
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            return "No optimization results available"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {},
            'detailed_results': self.optimization_results,
            'recommendations': []
        }
        
        # Find best configuration
        best_score = 0
        best_config = None
        
        for target_features, results in self.optimization_results.items():
            if results['best_f1_score'] > best_score:
                best_score = results['best_f1_score']
                best_config = results
        
        if best_config:
            report['optimization_summary'] = {
                'best_feature_count': best_config['target_features'],
                'best_model': best_config['best_model_name'],
                'best_f1_score': best_config['best_f1_score'],
                'feature_reduction_ratio': best_config['feature_reduction_ratio']
            }
            
            # Generate recommendations
            report['recommendations'] = [
                f"Use {best_config['target_features']} features for optimal performance",
                f"Deploy {best_config['best_model_name']} as the primary model",
                f"Feature reduction achieved: {best_config['feature_reduction_ratio']:.1f}%",
                f"Expected performance improvement: {best_score:.1%} F1 score"
            ]
        
        return report
    
    def save_optimization_results(self, filename="enhanced_optimization_results.json"):
        """Save optimization results to file"""
        try:
            report = self.generate_optimization_report()
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Optimization results saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            return False
    
    def visualize_optimization_results(self):
        """Visualize optimization results"""
        if not self.optimization_results:
            print("No results to visualize")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for visualization
            feature_counts = []
            f1_scores = []
            reduction_ratios = []
            
            for target_features, results in self.optimization_results.items():
                feature_counts.append(target_features)
                f1_scores.append(results['best_f1_score'])
                reduction_ratios.append(results['feature_reduction_ratio'])
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: F1 Score vs Feature Count
            ax1.plot(feature_counts, f1_scores, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Features')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('Performance vs Feature Count')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Feature Reduction Ratio
            ax2.bar(feature_counts, reduction_ratios, color='lightcoral', alpha=0.7)
            ax2.set_xlabel('Number of Features')
            ax2.set_ylabel('Feature Reduction (%)')
            ax2.set_title('Feature Reduction Ratio')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('enhanced_optimization_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Optimization results visualization saved as 'enhanced_optimization_results.png'")
            
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available for visualization")
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")

def main():
    """Main optimization pipeline"""
    print("🚀 ENHANCED PERFORMANCE OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    # Create sample data for demonstration
    print("1. Creating sample dataset...")
    np.random.seed(42)
    n_samples = 300
    n_features = 100
    
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
    X['feature_3'] = y * 3 + np.random.normal(0, 0.1, n_samples)
    
    print(f"   ✓ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize optimizer
    print("\n2. Initializing performance optimizer...")
    optimizer = EnhancedPerformanceOptimizer()
    
    # Run optimization
    print("\n3. Running performance optimization...")
    best_config, all_results = optimizer.optimize_performance(
        X, y, target_features_range=[15, 25, 35, 45], cv_folds=3
    )
    
    # Generate and save report
    print("\n4. Generating optimization report...")
    optimizer.save_optimization_results()
    
    # Create visualization
    print("\n5. Creating visualization...")
    optimizer.visualize_optimization_results()
    
    print("\n🎉 ENHANCED PERFORMANCE OPTIMIZATION COMPLETED!")
    print(f"🏆 Best configuration: {best_config['target_features']} features")
    print(f"🚀 Best performance: {best_config['best_f1_score']:.4f}")

if __name__ == "__main__":
    main()
