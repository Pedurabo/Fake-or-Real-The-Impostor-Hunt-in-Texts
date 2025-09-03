#!/usr/bin/env python3
"""
Test script for BERT pipeline
Verifies basic functionality before running full training
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_data_access():
    """Test if data can be accessed"""
    print("\nTesting data access...")
    
    # Check if data directories exist
    if not os.path.exists('data/train'):
        print("✗ data/train directory not found")
        return False
    
    if not os.path.exists('data/test'):
        print("✗ data/test directory not found")
        return False
    
    if not os.path.exists('data/train.csv'):
        print("✗ data/train.csv not found")
        return False
    
    print("✓ Data directories and files found")
    
    # Check if we can read some sample data
    try:
        import pandas as pd
        train_df = pd.read_csv('data/train.csv')
        print(f"✓ Training data loaded: {train_df.shape}")
        
        # Check if we can access some text files
        sample_article = f"data/train/article_{train_df.iloc[0]['id']:04d}"
        if os.path.exists(sample_article):
            text_files = [f for f in os.listdir(sample_article) if f.endswith('.txt')]
            if text_files:
                print(f"✓ Sample article found with {len(text_files)} text files")
            else:
                print("✗ Sample article has no text files")
                return False
        else:
            print("✗ Sample article directory not found")
            return False
            
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test if BERT models can be initialized"""
    print("\nTesting model initialization...")
    
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        
        # Test tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("✓ BERT tokenizer loaded")
        
        # Test model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        print("✓ BERT model loaded")
        
        # Test basic tokenization
        test_text = "This is a test sentence."
        encoding = tokenizer(test_text, return_tensors='pt')
        print(f"✓ Tokenization test passed: {encoding['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_device():
    """Test device availability"""
    print("\nTesting device...")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("ℹ Using CPU (CUDA not available)")
        
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("BERT PIPELINE TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_access,
        test_model_initialization,
        test_device
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! BERT pipeline is ready to run.")
        return True
    else:
        print("✗ Some tests failed. Please fix issues before running BERT pipeline.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
