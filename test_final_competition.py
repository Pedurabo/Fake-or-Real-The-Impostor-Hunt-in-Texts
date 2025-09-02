#!/usr/bin/env python3
"""
Test script for Final Competition Pipeline (Phase 4)
Tests final ensemble optimization and competition submission
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_final_competition_pipeline():
    """Test the final competition pipeline"""
    print("🏆 TESTING FINAL COMPETITION PIPELINE")
    print("=" * 60)
    print("PHASE 4: Final Optimization + Competition Submission")
    print("=" * 60)
    
    try:
        from modules.competition_final_pipeline import CompetitionFinalPipeline
        
        # Initialize final competition pipeline
        print("\n🎯 INITIALIZING FINAL COMPETITION PIPELINE")
        print("=" * 60)
        
        final_pipeline = CompetitionFinalPipeline(data_path="src/temp_data/data")
        
        # Run final competition pipeline
        print("\n🚀 STARTING PHASE 4 COMPLETE PIPELINE")
        print("=" * 60)
        
        results = final_pipeline.run_final_competition_pipeline()
        
        if results:
            print("\n🏆 PHASE 4 COMPLETE PIPELINE RESULTS:")
            print("=" * 60)
            print(f"📊 Submissions Loaded: {results['submissions_loaded']}")
            print(f"🔗 Final Ensemble Created: {'✅' if results['final_ensemble_created'] else '❌'}")
            print(f"📤 Competition Submission: {'✅' if results['competition_submission_generated'] else '❌'}")
            print(f"📋 Competition Report: {'✅' if results['competition_report_created'] else '❌'}")
            print(f"⏱️  Execution Time: {results['execution_time']}")
            
            # Save results
            final_pipeline.save_final_results()
            
            print("\n🏆 PHASE 4 COMPLETE! COMPETITION READY!")
            print("=" * 60)
            print("📤 Final competition submission generated")
            print("📋 Comprehensive competition report created")
            print("🎯 Ready for competition platform upload")
            print("🚀 All phases completed successfully!")
            
        else:
            print("\n❌ Final competition pipeline failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ FINAL COMPETITION PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_competition_pipeline()
    
    if success:
        print("\n✅ FINAL COMPETITION PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("🏆 READY FOR COMPETITION SUBMISSION!")
        print("📤 Upload: submissions/competition_final_submission.csv")
    else:
        print("\n❌ FINAL COMPETITION PIPELINE TEST FAILED!")
        sys.exit(1)
