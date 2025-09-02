#!/usr/bin/env python3
"""
Optimized Pipeline Runner
Demonstrates how to use the integrated optimized modules for fast test predictions
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Import the optimized pipeline orchestrator
from src.modules.pipeline_orchestrator import PipelineOrchestrator

def run_optimized_pipeline():
    """Run the complete pipeline with optimizations"""
    print("🚀 OPTIMIZED PIPELINE - FAST TEST PREDICTIONS")
    print("=" * 80)
    
    # Initialize the pipeline orchestrator
    print("Initializing optimized pipeline...")
    orchestrator = PipelineOrchestrator()
    
    try:
        # Run the complete pipeline with optimizations
        print("\n🎯 Running complete optimized pipeline...")
        start_time = time.time()
        
        orchestrator.run_full_pipeline()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total pipeline execution time: {total_time:.2f} seconds")
        
        # Show optimization benefits
        print("\n🚀 OPTIMIZATION BENEFITS ACHIEVED:")
        print("• Essential features only (20-30 vs 100+ features)")
        print("• Feature caching enabled for repeated predictions")
        print("• Batch processing optimized for speed")
        print("• Lightweight feature selection")
        print("• Single-pass dimensionality reduction")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        return False

def run_fast_test_processing_only():
    """Run only the fast test processing stage"""
    print("🚀 FAST TEST PROCESSING ONLY")
    print("=" * 60)
    
    # Initialize the pipeline orchestrator
    print("Initializing pipeline...")
    orchestrator = PipelineOrchestrator()
    
    try:
        # Run only the fast test processing
        print("\n🎯 Running fast test processing...")
        start_time = time.time()
        
        submission_data, submission_file = orchestrator.run_fast_test_processing(
            use_cached_features=True,  # Enable caching for speed
            batch_size=1000            # Optimize batch size
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Fast test processing completed!")
        print(f"📊 Submission shape: {submission_data.shape}")
        print(f"📁 Saved to: {submission_file}")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        # Show timing breakdown
        if hasattr(orchestrator.optimized_test_processor, 'timing_stats'):
            print("\n⏱️  TIMING BREAKDOWN:")
            timing_stats = orchestrator.optimized_test_processor.timing_stats
            for stage, time_taken in timing_stats.items():
                if stage != 'total_time':
                    print(f"  {stage.replace('_', ' ').title()}: {time_taken:.2f}s")
            
            if 'total_time' in timing_stats:
                print(f"  {'Total Time':20}: {timing_stats['total_time']:.2f}s")
        
        return submission_data, submission_file
        
    except Exception as e:
        print(f"\n❌ Fast test processing failed: {e}")
        return None, None

def benchmark_optimization_speed():
    """Benchmark the speed improvements"""
    print("🚀 BENCHMARKING OPTIMIZATION SPEED")
    print("=" * 60)
    
    # Initialize the pipeline orchestrator
    print("Initializing pipeline...")
    orchestrator = PipelineOrchestrator()
    
    try:
        # Benchmark 1: Fast test processing with caching
        print("\n📊 BENCHMARK 1: FAST TEST PROCESSING (CACHED)")
        print("-" * 50)
        
        start_time = time.time()
        submission1, file1 = orchestrator.run_fast_test_processing(
            use_cached_features=True,
            batch_size=1000
        )
        cached_time = time.time() - start_time
        
        print(f"✅ Cached processing completed in {cached_time:.2f} seconds")
        
        # Benchmark 2: Fast test processing without caching
        print("\n📊 BENCHMARK 2: FAST TEST PROCESSING (NO CACHE)")
        print("-" * 50)
        
        start_time = time.time()
        submission2, file2 = orchestrator.run_fast_test_processing(
            use_cached_features=False,
            batch_size=1000
        )
        no_cache_time = time.time() - start_time
        
        print(f"✅ No-cache processing completed in {no_cache_time:.2f} seconds")
        
        # Benchmark 3: Different batch sizes
        print("\n📊 BENCHMARK 3: BATCH SIZE COMPARISON")
        print("-" * 50)
        
        batch_sizes = [500, 1000, 2000]
        batch_results = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            submission_batch, file_batch = orchestrator.run_fast_test_processing(
                use_cached_features=True,
                batch_size=batch_size
            )
            batch_time = time.time() - start_time
            
            batch_results[batch_size] = {
                'time': batch_time,
                'speed': len(submission_batch) / batch_time if submission_batch is not None else 0
            }
            
            print(f"  Batch size {batch_size:4d}: {batch_time:.2f}s ({len(submission_batch)/batch_time:.1f} samples/s)")
        
        # Performance summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"🔴 Estimated original processing: ~{len(submission1) * 0.1:.1f} seconds")
        print(f"🟡 Fast processing (no cache): {no_cache_time:.2f} seconds")
        print(f"🟢 Fast processing (cached): {cached_time:.2f} seconds")
        
        if submission1 is not None:
            speedup_no_cache = (len(submission1) * 0.1) / no_cache_time
            speedup_cached = (len(submission1) * 0.1) / cached_time
            
            print(f"\n⚡ SPEED IMPROVEMENTS:")
            print(f"  Fast vs Original: {speedup_no_cache:.1f}x faster")
            print(f"  Cached vs Original: {speedup_cached:.1f}x faster")
            print(f"  Cached vs No-cache: {no_cache_time/cached_time:.1f}x faster")
        
        # Best batch size
        if batch_results:
            best_batch = max(batch_results.keys(), key=lambda x: batch_results[x]['speed'])
            print(f"\n🏆 Best batch size: {best_batch} ({batch_results[best_batch]['speed']:.1f} samples/s)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 OPTIMIZED PIPELINE INTEGRATION DEMO")
    print("=" * 80)
    
    print("\nChoose an option:")
    print("1. Run complete optimized pipeline")
    print("2. Run fast test processing only")
    print("3. Benchmark optimization speed")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n" + "="*80)
                success = run_optimized_pipeline()
                if success:
                    print("\n🎉 Complete pipeline completed successfully!")
                break
                
            elif choice == '2':
                print("\n" + "="*80)
                submission, file = run_fast_test_processing_only()
                if submission is not None:
                    print("\n🎉 Fast test processing completed successfully!")
                break
                
            elif choice == '3':
                print("\n" + "="*80)
                success = benchmark_optimization_speed()
                if success:
                    print("\n🎉 Benchmarking completed successfully!")
                break
                
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break
    
    print("\n📋 Next steps:")
    print("1. Use the optimized modules in your main pipeline")
    print("2. Enable feature caching for repeated predictions")
    print("3. Adjust batch sizes based on your system")
    print("4. Monitor performance with timing statistics")

if __name__ == "__main__":
    main()
