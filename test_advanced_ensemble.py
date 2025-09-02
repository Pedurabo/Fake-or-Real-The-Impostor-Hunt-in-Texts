#!/usr/bin/env python3
"""
Test script for Advanced Ensemble Pipeline (Phase 3)
Tests stacking, voting, and hybrid ensemble methods
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_advanced_ensemble_pipeline():
    """Test the advanced ensemble pipeline"""
    print("🚀 TESTING ADVANCED ENSEMBLE PIPELINE")
    print("=" * 60)
    print("PHASE 3: Stacking + Voting + Hybrid + Advanced Features")
    print("=" * 60)
    
    try:
        from modules.advanced_ensemble_pipeline import AdvancedEnsemblePipeline
        
        # Initialize advanced ensemble pipeline
        print("\n🎯 INITIALIZING ADVANCED ENSEMBLE PIPELINE")
        print("=" * 60)
        
        advanced_ensemble = AdvancedEnsemblePipeline(data_path="src/temp_data/data")
        
        # Run full advanced ensemble pipeline
        print("\n🚀 STARTING PHASE 3 COMPLETE PIPELINE")
        print("=" * 60)
        
        results = advanced_ensemble.run_full_advanced_pipeline()
        
        if results:
            print("\n🏆 PHASE 3 COMPLETE PIPELINE RESULTS:")
            print("=" * 60)
            print(f"📊 Base Pipelines: {'✅' if results['base_pipelines'] else '❌'}")
            print(f"🔬 Advanced Features: {'✅' if results['advanced_features'] else '❌'}")
            print(f"🏗️  Stacking Classifier: {'✅' if results['stacking_classifier'] else '❌'}")
            print(f"🗳️  Voting Classifier: {'✅' if results['voting_classifier'] else '❌'}")
            print(f"🔗 Hybrid Predictions: {'✅' if results['hybrid_predictions'] else '❌'}")
            print(f"📤 Advanced Submission: {'✅' if results['submission_generated'] else '❌'}")
            print(f"⏱️  Execution Time: {results['execution_time']}")
            
            # Save results
            advanced_ensemble.save_advanced_results()
            
            print("\n🏆 PHASE 3 COMPLETE! READY FOR COMPETITION!")
            print("=" * 60)
            print("📤 Advanced ensemble submission generated")
            print("🔬 Stacking + Voting + Hybrid ensemble created")
            print("🎯 Ready to upload to competition leaderboard")
            
        else:
            print("\n❌ Advanced ensemble pipeline failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ ADVANCED ENSEMBLE PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_advanced_ensemble_pipeline()
    
    if success:
        print("\n✅ ADVANCED ENSEMBLE PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for Phase 3 competition submission!")
    else:
        print("\n❌ ADVANCED ENSEMBLE PIPELINE TEST FAILED!")
        sys.exit(1)
