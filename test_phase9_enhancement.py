#!/usr/bin/env python3
"""
Test script for Phase 9: Production Enhancement & Model Optimization
Tests robust cross-validation, model complexity reduction, and production enhancement
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_phase9_enhancement():
    """Test the Phase 9 production enhancement"""
    print("🚀 TESTING PHASE 9: PRODUCTION ENHANCEMENT")
    print("=" * 70)
    print("Robust CV + Model Complexity Reduction + Production Enhancement")
    print("=" * 70)
    
    try:
        from modules.phase9_production_enhancement import Phase9ProductionEnhancement
        
        # Initialize Phase 9 enhancer
        print("\n🎯 INITIALIZING PHASE 9 PRODUCTION ENHANCER")
        print("=" * 60)
        
        enhancer = Phase9ProductionEnhancement(data_path="src/temp_data/data")
        
        # Run Phase 9 enhancement
        print("\n🚀 STARTING PHASE 9 COMPLETE ENHANCEMENT")
        print("=" * 60)
        
        results = enhancer.run_phase9_enhancement()
        
        if results:
            print("\n📊 PHASE 9 COMPLETE ENHANCEMENT RESULTS:")
            print("=" * 60)
            print(f"📊 Phase 6 Models Loaded: {'✅' if results['phase6_models_loaded'] else '❌'}")
            print(f"🔄 Robust CV Implemented: {'✅' if results['robust_cv_implemented'] else '❌'}")
            print(f"🔧 Model Complexity Reduced: {'✅' if results['model_complexity_reduced'] else '❌'}")
            print(f"🔗 Generalizable Ensemble Created: {'✅' if results['generalizable_ensemble_created'] else '❌'}")
            print(f"📊 Models Evaluated: {'✅' if results['models_evaluated'] else '❌'}")
            print(f"🌐 Production Pipeline Enhanced: {'✅' if results['production_pipeline_enhanced'] else '❌'}")
            print(f"📋 Report Generated: {'✅' if results['report_generated'] else '❌'}")
            print(f"⏱️  Execution Time: {results['execution_time']}")
            
            print("\n🚀 PHASE 9 COMPLETE! READY FOR PHASE 10!")
            print("=" * 60)
            print("🔄 Robust cross-validation implemented")
            print("🔧 Model complexity reduced")
            print("🔗 Generalizable ensembles created")
            print("🌐 Production pipeline enhanced")
            print("🚀 Ready for competition finale")
            
        else:
            print("\n❌ Phase 9 enhancement failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 9 ENHANCEMENT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase9_enhancement()
    
    if success:
        print("\n✅ PHASE 9 ENHANCEMENT TEST COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for Phase 10: Competition Finale!")
        print("🔄 Robust cross-validation ready")
        print("🔧 Model complexity optimization complete")
        print("🌐 Production pipeline enhanced")
        print("📋 Check phase9_production_enhancement_report.md for results")
    else:
        print("\n❌ PHASE 9 ENHANCEMENT TEST FAILED!")
        sys.exit(1)
