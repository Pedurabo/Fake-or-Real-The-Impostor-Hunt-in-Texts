#!/usr/bin/env python3
"""
Test script for Phase 5: Competition Performance Analysis
Tests submission analysis and improvement planning
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_phase5_analysis():
    """Test the Phase 5 analysis pipeline"""
    print("📊 TESTING PHASE 5: COMPETITION PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Performance Analysis + Improvement Planning")
    print("=" * 60)
    
    try:
        from modules.competition_performance_analyzer import CompetitionPerformanceAnalyzer
        
        # Initialize Phase 5 analyzer
        print("\n🎯 INITIALIZING PHASE 5 ANALYZER")
        print("=" * 60)
        
        analyzer = CompetitionPerformanceAnalyzer(data_path="src/temp_data/data")
        
        # Run Phase 5 analysis
        print("\n🚀 STARTING PHASE 5 COMPLETE ANALYSIS")
        print("=" * 60)
        
        results = analyzer.run_phase5_analysis()
        
        if results:
            print("\n📊 PHASE 5 COMPLETE ANALYSIS RESULTS:")
            print("=" * 60)
            print(f"📊 Submission Loaded: {'✅' if results['submission_loaded'] else '❌'}")
            print(f"🔍 Pattern Analysis: {'✅' if results['pattern_analysis_completed'] else '❌'}")
            print(f"🎯 Improvement Plan: {'✅' if results['improvement_plan_generated'] else '❌'}")
            print(f"📋 Performance Report: {'✅' if results['performance_report_created'] else '❌'}")
            print(f"⏱️  Execution Time: {results['execution_time']}")
            
            # Save results
            analyzer.save_phase5_results()
            
            print("\n🚀 PHASE 5 COMPLETE! READY FOR PHASE 6!")
            print("=" * 60)
            print("📊 Competition performance analyzed")
            print("🎯 Improvement strategies generated")
            print("📋 Comprehensive report created")
            print("🚀 Ready for advanced model optimization")
            
        else:
            print("\n❌ Phase 5 analysis failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 5 ANALYSIS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase5_analysis()
    
    if success:
        print("\n✅ PHASE 5 ANALYSIS TEST COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for Phase 6: Advanced Model Optimization!")
        print("📋 Check phase5_performance_report.md for detailed analysis")
    else:
        print("\n❌ PHASE 5 ANALYSIS TEST FAILED!")
        sys.exit(1)
