#!/usr/bin/env python3
"""
Test script for the Ensemble Pipeline
Demonstrates the complete Phase 2: Fast + Transformer + Ensemble
"""

import sys
import os
import time

# Add src to path
sys.path.append('src')

from modules.ensemble_pipeline import EnsemblePipeline

def main():
    """Test the complete ensemble pipeline"""
    print("🚀 TESTING COMPLETE ENSEMBLE PIPELINE")
    print("=" * 60)
    print("PHASE 2: Fast Models + Transformer Models + Ensemble")
    print("=" * 60)
    
    # Initialize ensemble pipeline
    ensemble = EnsemblePipeline(data_path="src/temp_data/data")
    
    # Run full ensemble pipeline and measure time
    start_time = time.time()
    
    try:
        print("\n🎯 STARTING PHASE 2 COMPLETE PIPELINE")
        print("=" * 60)
        
        # Run the complete ensemble pipeline
        results = ensemble.run_full_ensemble_pipeline()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results:
            print(f"\n⏱️  TOTAL EXECUTION TIME: {total_time:.2f} seconds")
            print(f"⏱️  TOTAL EXECUTION TIME: {total_time/60:.2f} minutes")
            
            # Print comprehensive results
            print(f"\n🏆 PHASE 2 COMPLETE PIPELINE RESULTS:")
            print("=" * 60)
            print(f"📊 Fast Pipeline: {'✅ SUCCESS' if results['fast_pipeline'] else '❌ FAILED'}")
            print(f"🤖 Transformer Pipeline: {'✅ SUCCESS' if results['transformer_pipeline'] else '❌ FAILED'}")
            print(f"🔗 Ensemble Created: {'✅ SUCCESS' if results['ensemble_created'] else '❌ FAILED'}")
            print(f"📤 Submission Generated: {'✅ SUCCESS' if results['submission_generated'] else '❌ FAILED'}")
            print(f"⏱️  Total Execution Time: {results['execution_time']}")
            
            # Performance summary
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print("=" * 60)
            
            if results['fast_pipeline'] and hasattr(ensemble.fast_pipeline, 'best_score'):
                print(f"🏃 Fast Models Accuracy: {ensemble.fast_pipeline.best_score:.4f}")
            
            if results['transformer_pipeline'] and hasattr(ensemble.transformer_pipeline, 'accuracy'):
                print(f"🤖 Transformer Accuracy: {ensemble.transformer_pipeline.accuracy:.4f}")
            
            if results['ensemble_created'] and hasattr(ensemble, 'ensemble_predictions'):
                print(f"🔗 Ensemble Articles: {len(ensemble.ensemble_predictions)}")
                
                # Show ensemble method distribution
                if 'ensemble_method' in ensemble.ensemble_predictions.columns:
                    method_counts = ensemble.ensemble_predictions['ensemble_method'].value_counts()
                    print(f"📊 Ensemble Method Distribution:")
                    for method, count in method_counts.items():
                        percentage = (count / len(ensemble.ensemble_predictions)) * 100
                        print(f"   • {method}: {count} articles ({percentage:.1f}%)")
            
            # Save results
            ensemble.save_ensemble_results()
            
            print(f"\n💾 Results saved to ensemble_results.json")
            
            # Competition readiness
            print(f"\n🏆 COMPETITION READINESS:")
            print("=" * 60)
            if results['submission_generated']:
                print("✅ READY FOR COMPETITION SUBMISSION!")
                print("📤 Submission files generated in submissions/ directory")
                print("🚀 Ready to upload to competition leaderboard")
            else:
                print("⚠️  SUBMISSION NOT GENERATED")
                print("🔧 Check pipeline execution for errors")
            
        else:
            print(f"\n❌ ENSEMBLE PIPELINE FAILED")
            print("🔧 Check individual pipeline components")
            
    except Exception as e:
        print(f"\n❌ ENSEMBLE PIPELINE EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 PHASE 2 COMPLETE PIPELINE TEST FINISHED")
    print("=" * 60)

if __name__ == "__main__":
    main()
