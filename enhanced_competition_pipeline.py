#!/usr/bin/env python3
"""
Enhanced Competition Pipeline
Complete pipeline integrating enhanced feature selection and model training for competition
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

class EnhancedCompetitionPipeline:
    """
    Complete enhanced pipeline for competition submission
    """
    
    def __init__(self):
        self.feature_selector = EnhancedFeatureSelector()
        self.model_trainer = EnhancedModelTrainer()
        self.best_model = None
        self.selected_features = None
        self.pipeline_results = {}
        
    def run_complete_pipeline(self, train_data, test_data, target_features=30, cv_folds=5):
        """
        Run complete enhanced pipeline
        """
        print("🚀 ENHANCED COMPETITION PIPELINE")
        print("=" * 70)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data preparation
            print("\n1. 📊 DATA PREPARATION")
            X_train, y_train = self._prepare_training_data(train_data)
            X_test = self._prepare_test_data(test_data)
            
            print(f"   ✓ Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"   ✓ Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            # Step 2: Enhanced feature selection
            print(f"\n2. 🔬 ENHANCED FEATURE SELECTION ({target_features} features)")
            X_train_selected, selected_features = self.feature_selector.maximize_performance_selection(
                X_train, y_train, target_features=target_features, cv_folds=cv_folds
            )
            
            # Apply same selection to test data
            X_test_selected = X_test[selected_features]
            
            self.selected_features = selected_features
            print(f"   ✓ Selected {len(selected_features)} features")
            print(f"   ✓ Feature reduction: {((X_train.shape[1] - len(selected_features)) / X_train.shape[1] * 100):.1f}%")
            
            # Step 3: Data splitting for validation
            print(f"\n3. 📈 DATA SPLITTING FOR VALIDATION")
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_selected, y_train, test_size=0.3, random_state=42, stratify=y_train
            )
            
            print(f"   ✓ Training split: {X_train_split.shape[0]} samples")
            print(f"   ✓ Validation split: {X_val_split.shape[0]} samples")
            
            # Step 4: Enhanced model training
            print(f"\n4. 🎯 ENHANCED MODEL TRAINING")
            best_model, best_score = self.model_trainer.train_enhanced_models(
                X_train_split, X_val_split, y_train_split, y_val_split, selected_features
            )
            
            self.best_model = best_model
            
            # Step 5: Final training on full selected data
            print(f"\n5. 🚀 FINAL MODEL TRAINING ON FULL DATA")
            print("   • Training best model on full selected training data...")
            self.best_model.fit(X_train_selected, y_train)
            
            # Step 6: Generate predictions
            print(f"\n6. 🔮 GENERATING PREDICTIONS")
            test_predictions = self.best_model.predict(X_test_selected)
            test_probabilities = self.best_model.predict_proba(X_test_selected) if hasattr(self.best_model, 'predict_proba') else None
            
            # Step 7: Create submission
            print(f"\n7. 📝 CREATING SUBMISSION")
            submission = self._create_submission(test_data, test_predictions, test_probabilities)
            
            # Step 8: Generate comprehensive report
            print(f"\n8. 📊 GENERATING COMPREHENSIVE REPORT")
            self._generate_comprehensive_report(
                X_train_selected, y_train, X_val_split, y_val_split,
                best_score, selected_features, start_time
            )
            
            # Store results
            self.pipeline_results = {
                'selected_features': selected_features,
                'best_model_name': self.model_trainer.best_model_name,
                'best_score': best_score,
                'feature_reduction_ratio': ((X_train.shape[1] - len(selected_features)) / X_train.shape[1] * 100),
                'submission_shape': submission.shape,
                'pipeline_duration': str(datetime.now() - start_time)
            }
            
            print(f"\n✅ ENHANCED COMPETITION PIPELINE COMPLETED!")
            print(f"🏆 Best model: {self.model_trainer.best_model_name}")
            print(f"🚀 Best F1 Score: {best_score:.4f}")
            print(f"📊 Selected features: {len(selected_features)}")
            print(f"⏱️ Pipeline duration: {self.pipeline_results['pipeline_duration']}")
            
            return submission, self.pipeline_results
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _prepare_training_data(self, train_data):
        """Prepare training data"""
        # Remove non-feature columns
        feature_cols = [col for col in train_data.columns if col not in ['id', 'real_text_id']]
        X = train_data[feature_cols]
        y = train_data['real_text_id']
        
        return X, y
    
    def _prepare_test_data(self, test_data):
        """Prepare test data"""
        # Remove non-feature columns
        feature_cols = [col for col in test_data.columns if col not in ['id']]
        X = test_data[feature_cols]
        
        return X
    
    def _create_submission(self, test_data, predictions, probabilities):
        """Create competition submission"""
        submission = test_data[['id']].copy()
        submission['real_text_id'] = predictions
        
        # Add confidence scores if available
        if probabilities is not None:
            submission['confidence'] = np.max(probabilities, axis=1)
        
        return submission
    
    def _generate_comprehensive_report(self, X_train, y_train, X_val, y_val, best_score, selected_features, start_time):
        """Generate comprehensive pipeline report"""
        report = {
            'pipeline_info': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_duration': str(datetime.now() - start_time),
                'target_features': len(selected_features),
                'original_features': X_train.shape[1],
                'feature_reduction_ratio': ((X_train.shape[1] - len(selected_features)) / X_train.shape[1] * 100)
            },
            'model_performance': {
                'best_model_name': self.model_trainer.best_model_name,
                'best_f1_score': best_score,
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0]
            },
            'feature_selection': {
                'selected_features': selected_features,
                'feature_importance': self._get_feature_importance(X_train, y_train, selected_features)
            },
            'training_summary': self.model_trainer.get_training_summary()
        }
        
        # Save report
        with open('enhanced_pipeline_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("   ✓ Comprehensive report saved as 'enhanced_pipeline_report.json'")
        
        # Save selected features
        features_df = pd.DataFrame({
            'feature_name': selected_features,
            'importance_rank': range(1, len(selected_features) + 1)
        })
        features_df.to_csv('selected_features_enhanced.csv', index=False)
        print("   ✓ Selected features saved as 'selected_features_enhanced.csv'")
    
    def _get_feature_importance(self, X, y, selected_features):
        """Get feature importance for selected features"""
        try:
            # Use Random Forest for feature importance
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance_dict = {}
            for i, feature in enumerate(selected_features):
                if feature in X.columns:
                    feature_idx = X.columns.get_loc(feature)
                    importance_dict[feature] = rf.feature_importances_[feature_idx]
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance)
            
        except Exception as e:
            print(f"   ⚠️ Could not calculate feature importance: {str(e)}")
            return {}
    
    def save_best_model(self, filename="enhanced_best_model.pkl"):
        """Save the best trained model"""
        try:
            import joblib
            joblib.dump(self.best_model, filename)
            print(f"✅ Best model saved as '{filename}'")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_best_model(self, filename="enhanced_best_model.pkl"):
        """Load a saved model"""
        try:
            import joblib
            self.best_model = joblib.load(filename)
            print(f"✅ Model loaded from '{filename}'")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

def main():
    """Main pipeline execution"""
    print("🚀 ENHANCED COMPETITION PIPELINE EXECUTION")
    print("=" * 70)
    
    # Create sample data for demonstration
    print("1. Creating sample competition dataset...")
    np.random.seed(42)
    n_train_samples = 500
    n_test_samples = 200
    n_features = 100
    
    # Training data
    train_data = pd.DataFrame(
        np.random.randn(n_train_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    train_data['id'] = range(n_train_samples)
    train_data['real_text_id'] = np.random.choice([1, 2], size=n_train_samples)
    
    # Test data
    test_data = pd.DataFrame(
        np.random.randn(n_test_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    test_data['id'] = range(n_test_samples)
    
    # Make some features predictive
    for data in [train_data, test_data]:
        data['feature_0'] = np.random.choice([1, 2], size=len(data)) + np.random.normal(0, 0.1, len(data))
        data['feature_1'] = np.random.choice([1, 2], size=len(data)) * 2 + np.random.normal(0, 0.1, len(data))
        data['feature_2'] = (np.random.choice([1, 2], size=len(data)) == 1).astype(int) + np.random.normal(0, 0.1, len(data))
    
    print(f"   ✓ Training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"   ✓ Test data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
    
    # Initialize pipeline
    print("\n2. Initializing enhanced competition pipeline...")
    pipeline = EnhancedCompetitionPipeline()
    
    # Run complete pipeline
    print("\n3. Running complete enhanced pipeline...")
    submission, results = pipeline.run_complete_pipeline(
        train_data, test_data, target_features=25, cv_folds=3
    )
    
    if submission is not None:
        # Save submission
        submission.to_csv('enhanced_competition_submission.csv', index=False)
        print(f"\n✅ Submission saved as 'enhanced_competition_submission.csv'")
        
        # Save best model
        pipeline.save_best_model()
        
        print(f"\n🎉 ENHANCED COMPETITION PIPELINE SUCCESSFULLY COMPLETED!")
        print(f"📊 Submission shape: {submission.shape}")
        print(f"🏆 Best model: {results['best_model_name']}")
        print(f"🚀 Best score: {results['best_score']:.4f}")
    else:
        print(f"\n💥 Pipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
