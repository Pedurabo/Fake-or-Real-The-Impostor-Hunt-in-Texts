#!/usr/bin/env python3
"""
Simple test for Transformer Pipeline imports
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if we can import the transformer pipeline"""
    try:
        print("🔍 Testing imports...")
        
        # Test basic imports
        import torch
        print("  ✅ PyTorch imported successfully")
        
        import transformers
        print(f"  ✅ Transformers imported successfully (version: {transformers.__version__})")
        
        # Test our pipeline
        from modules.transformer_pipeline import TransformerPipeline
        print("  ✅ TransformerPipeline imported successfully")
        
        print("\n🎉 All imports successful! Transformer pipeline is ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic pipeline functionality"""
    try:
        print("\n🔧 Testing basic functionality...")
        
        from modules.transformer_pipeline import TransformerPipeline
        
        # Initialize pipeline
        pipeline = TransformerPipeline(
            model_name="distilbert-base-uncased",
            data_path="src/temp_data/data"
        )
        
        print("  ✅ Pipeline initialized successfully")
        print(f"  ✅ Device: {pipeline.device}")
        print(f"  ✅ Model name: {pipeline.model_name}")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests"""
    print("🚀 SIMPLE TRANSFORMER PIPELINE TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎯 TRANSFORMER PIPELINE READY FOR PHASE 2!")
            print("=" * 50)
            print("✅ All tests passed")
            print("🚀 Ready to run full transformer pipeline")
        else:
            print("\n⚠️  Basic functionality issues detected")
    else:
        print("\n❌ Import issues detected")

if __name__ == "__main__":
    main()
