#!/usr/bin/env python3
"""
Test script for Phase 7: Production Pipeline
Tests model serving, API development, and deployment scripts
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_phase7_production():
    """Test the Phase 7 production pipeline"""
    print("🚀 TESTING PHASE 7: PRODUCTION PIPELINE")
    print("=" * 60)
    print("Model Serving + API Development + Deployment Scripts")
    print("=" * 60)
    
    try:
        from modules.production_pipeline import ProductionPipeline
        
        # Initialize Phase 7 production pipeline
        print("\n🎯 INITIALIZING PHASE 7 PRODUCTION PIPELINE")
        print("=" * 60)
        
        production = ProductionPipeline(data_path="src/temp_data/data")
        
        # Run Phase 7 production setup
        print("\n🚀 STARTING PHASE 7 COMPLETE PRODUCTION SETUP")
        print("=" * 60)
        
        results = production.run_phase7_production()
        
        if results:
            print("\n📊 PHASE 7 COMPLETE PRODUCTION RESULTS:")
            print("=" * 60)
            print(f"📊 Models Loaded: {'✅' if results['models_loaded'] else '❌'}")
            print(f"🌐 API Created: {'✅' if results['api_created'] else '❌'}")
            print(f"📜 Scripts Created: {'✅' if results['scripts_created'] else '❌'}")
            print(f"📋 Report Generated: {'✅' if results['report_generated'] else '❌'}")
            print(f"⏱️  Execution Time: {results['execution_time']}")
            
            # Save results
            production.save_phase7_results()
            
            print("\n🚀 PHASE 7 COMPLETE! READY FOR PHASE 8!")
            print("=" * 60)
            print("🌐 Production API created")
            print("📜 Deployment scripts generated")
            print("📋 Comprehensive documentation completed")
            print("🚀 Ready for competition finale")
            
        else:
            print("\n❌ Phase 7 production failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 7 PRODUCTION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase7_production()
    
    if success:
        print("\n✅ PHASE 7 PRODUCTION TEST COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for Phase 8: Competition Finale!")
        print("📋 Check phase7_production_report.md for deployment details")
        print("🌐 Production API ready for deployment")
    else:
        print("\n❌ PHASE 7 PRODUCTION TEST FAILED!")
        sys.exit(1)
