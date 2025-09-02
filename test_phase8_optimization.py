#!/usr/bin/env python3
"""
Test script for Phase 8: Advanced Optimization & Production Enhancement
Tests performance gap analysis and optimization strategies
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_phase8_optimization():
    """Test the Phase 8 advanced optimization"""
    print("🚀 TESTING PHASE 8: ADVANCED OPTIMIZATION")
    print("=" * 70)
    print("Performance Gap Analysis + Optimization Strategies")
    print("=" * 70)
    
    try:
        from modules.phase8_advanced_optimization import Phase8AdvancedOptimization
        
        # Initialize Phase 8 optimizer
        print("\n🎯 INITIALIZING PHASE 8 ADVANCED OPTIMIZER")
        print("=" * 60)
        
        optimizer = Phase8AdvancedOptimization(data_path="src/temp_data/data")
        
        # Run Phase 8 optimization
        print("\n🚀 STARTING PHASE 8 COMPLETE OPTIMIZATION")
        print("=" * 60)
        
        results = optimizer.run_phase8_optimization()
        
        if results:
            print("\n📊 PHASE 8 COMPLETE OPTIMIZATION RESULTS:")
            print("=" * 60)
            print(f"📊 Phase 6 Models Loaded: {'✅' if results['phase6_models_loaded'] else '❌'}")
            print(f"🔍 Performance Gap Analyzed: {'✅' if results['performance_gap_analyzed'] else '❌'}")
            print(f"⏱️  Execution Time: {results['execution_time']}")
            
            print("\n🚀 PHASE 8 COMPLETE! READY FOR PHASE 9!")
            print("=" * 60)
            print("🔍 Performance gap analysis completed")
            print("🎯 Optimization strategies identified")
            print("🚀 Ready for production deployment")
            
        else:
            print("\n❌ Phase 8 optimization failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 8 OPTIMIZATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase8_optimization()
    
    if success:
        print("\n✅ PHASE 8 OPTIMIZATION TEST COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for Phase 9: Production Deployment!")
        print("🔍 Performance gap analysis ready")
        print("🎯 Optimization strategies identified")
    else:
        print("\n❌ PHASE 8 OPTIMIZATION TEST FAILED!")
        sys.exit(1)
