#!/usr/bin/env python3
"""
Test script for the Simplified Transformer Pipeline
Demonstrates basic transformer functionality without compatibility issues
"""

import sys
import os
import time

# Add src to path
sys.path.append('src')

from modules.transformer_pipeline_simple import SimpleTransformerPipeline

def main():
    """Test the simplified transformer pipeline"""
    print("🚀 TESTING SIMPLIFIED TRANSFORMER PIPELINE")
    print("=" * 60)
    print("PHASE 2: Simplified Transformer Models")
    print("=" * 60)
    
    # Initialize pipeline with a smaller model for testing
    pipeline = SimpleTransformerPipeline(
        model_name="distilbert-base-uncased",  # Smaller, faster model for testing
        data_path="src/temp_data/data"
    )
    
    # Run pipeline and measure time
    start_time = time.time()
    
    try:
        print("\n📊 PHASE 2: SIMPLIFIED TRANSFORMER MODELS")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        print("\n📊 STEP 1: DATA LOADING AND PREPARATION")
        pipeline.load_and_prepare_data()
        
        # Step 2: Initialize model
        print("\n🤖 STEP 2: MODEL INITIALIZATION")
        pipeline.initialize_model()
        
        # Step 3: Create datasets
        print("\n📚 STEP 3: DATASET CREATION")
        pipeline.create_datasets()
        
        # Step 4: Train model (reduced epochs for testing)
        print("\n🏋️  STEP 4: MODEL TRAINING")
        pipeline.train_model(epochs=2, batch_size=4)  # Reduced for testing
        
        # Step 5: Evaluate model
        print("\n📈 STEP 5: MODEL EVALUATION")
        accuracy = pipeline.evaluate_model()
        
        # Step 6: Generate test predictions
        print("\n🧪 STEP 6: TEST PREDICTIONS")
        test_results = pipeline.predict_test_data()
        
        # Step 7: Save model
        print("\n💾 STEP 7: MODEL SAVING")
        pipeline.save_model()
        
        # Step 8: Generate submission
        print("\n📤 STEP 8: SUBMISSION GENERATION")
        submission = pipeline.generate_submission()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print(f"⏱️  TOTAL EXECUTION TIME: {total_time/60:.2f} minutes")
        
        # Print results summary
        print(f"\n🏆 SIMPLIFIED TRANSFORMER PIPELINE RESULTS:")
        print(f"📊 ACCURACY: {accuracy:.4f}")
        print(f"🧪 TEST ARTICLES: {len(test_results)}")
        print(f"📤 SUBMISSION READY: {submission is not None}")
        
        if submission is not None:
            print(f"\n📋 SUBMISSION PREVIEW:")
            print(submission.head())
            
            # Save results summary
            results_summary = {
                'pipeline_type': 'simple_transformer',
                'model_name': pipeline.model_name,
                'accuracy': accuracy,
                'execution_time_seconds': total_time,
                'test_articles_processed': len(test_results),
                'submission_generated': submission is not None
            }
            
            print(f"\n💾 Results summary saved to simple_transformer_results.json")
            
    except Exception as e:
        print(f"\n❌ SIMPLIFIED TRANSFORMER PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
