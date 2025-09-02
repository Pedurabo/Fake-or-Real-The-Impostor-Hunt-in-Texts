#!/usr/bin/env python3
"""
Test script for the optimized pipeline
Demonstrates fast training using clustering and optimized algorithms
"""

import sys
import os
import time

# Add src to path
sys.path.append('src')

from modules.optimized_pipeline_orchestrator import OptimizedPipelineOrchestrator

def main():
    """Run the optimized pipeline"""
    print("🚀 TESTING OPTIMIZED PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = OptimizedPipelineOrchestrator(data_path="src/temp_data/data")
    
    # Run pipeline and measure time
    start_time = time.time()
    
    try:
        pipeline.run_optimized_pipeline()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
        print(f"⏱️  TOTAL EXECUTION TIME: {total_time/60:.2f} minutes")
        
        # Get results
        best_model = pipeline.get_best_model()
        performances = pipeline.get_model_performances()
        test_predictions = pipeline.get_test_predictions()
        
        print(f"\n🏆 BEST MODEL: {best_model.__class__.__name__}")
        print(f"📊 ACCURACY: {pipeline.best_score:.4f}")
        
        print(f"\n📈 MODEL PERFORMANCES:")
        for model_name, perf in performances.items():
            print(f"  • {model_name}: {perf['accuracy']:.4f}")
        
        if test_predictions is not None:
            print(f"\n🧪 TEST PREDICTIONS: {len(test_predictions)} articles processed")
            print(test_predictions.head())
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
