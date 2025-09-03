#!/usr/bin/env python3
"""
Run Complete Pipeline with Kernel Methods
This script runs the entire pipeline from start to finish
"""

import sys
import os
sys.path.append('src')

def main():
    print("🚀 RUNNING COMPLETE PIPELINE WITH KERNEL METHODS")
    print("=" * 60)
    
    try:
        from modules.pipeline_orchestrator import PipelineOrchestrator
        
        # Initialize pipeline
        print("📊 Initializing pipeline...")
        pipeline = PipelineOrchestrator()
        
        # Run the complete pipeline
        print("\n🚀 STARTING COMPLETE PIPELINE...")
        print("=" * 60)
        
        pipeline.run_full_pipeline()
        
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show kernel results
        if hasattr(pipeline, 'pipeline_results') and 'kernel_feature_transformation' in pipeline.pipeline_results:
            kernel_results = pipeline.pipeline_results['kernel_feature_transformation']
            print("\n🏆 KERNEL TRANSFORMATION RESULTS:")
            print("-" * 40)
            print(f"Best Kernel Method: {kernel_results['best_kernel_method'].upper()}")
            print(f"Original Features: {kernel_results['original_features']}")
            print(f"Kernel Features: {kernel_results['kernel_features']}")
            print(f"Reduced Features: {kernel_results['reduced_kernel_features']}")
            print(f"Expansion Ratio: {kernel_results['expansion_ratio']:.2f}x")
            
            print("\n📊 All Kernel Method Performance:")
            for kernel, ratio in kernel_results['all_kernel_results'].items():
                print(f"• {kernel.upper()}: {ratio:.2f}x expansion")
        
        # Show overall results
        print("\n📊 OVERALL PIPELINE RESULTS:")
        print("-" * 40)
        for stage, status in pipeline.stages.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {stage.replace('_', ' ').title()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
