#!/usr/bin/env python3
"""
Working Dataset Downloader - Using Actually Available Datasets
Tested alternatives that work with current HuggingFace
"""

from pathlib import Path
import sys

print("="*70)
print("WORKING DATASET DOWNLOADER")
print("Using verified, working datasets")
print("="*70)
print()

# Check datasets package
try:
    from datasets import load_dataset
    import datasets
    print(f"✓ datasets package version: {datasets.__version__}")
except ImportError:
    print("✗ datasets package required")
    print("Install: pip install datasets")
    sys.exit(1)

print()
print("="*70)
print("AVAILABLE DATASETS (All Tested & Working)")
print("="*70)
print()
print("1. Code Alpaca - Python code instructions")
print("   • 20K instruction-code pairs")
print("   • Perfect for your code generation task")
print("   • Size: ~50MB, Time: 1-2 min")
print()
print("2. Python Code Instructions")
print("   • Iamtarun/python_code_instructions_18k_alpaca")
print("   • 18K examples")
print("   • Size: ~30MB, Time: 1 min")
print()
print("3. MBPP - Python code problems")
print("   • google-research-datasets/mbpp")
print("   • 1,000 programming problems")
print("   • Size: ~5MB, Time: <1 min")
print()
print("4. Download ALL (Options 1+2+3)")
print("   • Total: ~35K+ examples")
print("   • Combined ~85MB")
print()
print("="*70)

choice = input("\nWhat to download? (1-4): ").strip()

results = {}

def download_code_alpaca():
    """Download Code Alpaca dataset"""
    print("\n" + "="*70)
    print("DOWNLOADING: Code Alpaca")
    print("="*70)
    
    try:
        print("Loading dataset...")
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        
        print(f"✓ Loaded: {len(dataset)} examples")
        
        # Save
        save_path = "./data/raw/code_alpaca"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to: {save_path}")
        dataset.save_to_disk(save_path)
        
        # Sample
        sample = dataset[0]
        print("\nSample:")
        print(f"  Instruction: {sample.get('instruction', '')[:100]}...")
        print(f"  Output: {sample.get('output', '')[:100]}...")
        
        print("\n✓ Code Alpaca downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def download_python_instructions():
    """Download Python code instructions"""
    print("\n" + "="*70)
    print("DOWNLOADING: Python Code Instructions")
    print("="*70)
    
    try:
        print("Loading dataset...")
        dataset = load_dataset("Iamtarun/python_code_instructions_18k_alpaca", split="train")
        
        print(f"✓ Loaded: {len(dataset)} examples")
        
        # Save
        save_path = "./data/raw/python_instructions"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to: {save_path}")
        dataset.save_to_disk(save_path)
        
        # Sample
        sample = dataset[0]
        print("\nSample:")
        print(f"  Instruction: {sample.get('instruction', sample.get('prompt', ''))[:100]}...")
        
        print("\n✓ Python Instructions downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def download_mbpp():
    """Download MBPP (Python programming problems)"""
    print("\n" + "="*70)
    print("DOWNLOADING: MBPP (Python Problems)")
    print("="*70)
    
    try:
        print("Loading dataset...")
        dataset = load_dataset("mbpp", "sanitized", split="train")
        
        print(f"✓ Loaded: {len(dataset)} examples")
        
        # Save
        save_path = "./data/raw/mbpp"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to: {save_path}")
        dataset.save_to_disk(save_path)
        
        # Sample
        sample = dataset[0]
        print("\nSample:")
        print(f"  Task: {sample.get('text', '')[:100]}...")
        print(f"  Code: {sample.get('code', '')[:100]}...")
        
        print("\n✓ MBPP downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

# Download based on choice
if choice == '1':
    results['Code Alpaca'] = download_code_alpaca()
elif choice == '2':
    results['Python Instructions'] = download_python_instructions()
elif choice == '3':
    results['MBPP'] = download_mbpp()
elif choice == '4':
    results['Code Alpaca'] = download_code_alpaca()
    results['Python Instructions'] = download_python_instructions()
    results['MBPP'] = download_mbpp()
else:
    print("Invalid choice")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("DOWNLOAD SUMMARY")
print("="*70)

for dataset, success in results.items():
    status = "✓ Success" if success else "✗ Failed"
    print(f"  {status} {dataset}")

successful = sum(results.values())
total = len(results)

print(f"\nSuccessfully downloaded: {successful}/{total}")

# Complete inventory
print("\n" + "="*70)
print("COMPLETE DATASET INVENTORY")
print("="*70)

print("\n✅ Previously Downloaded:")
print("  ✓ Stack Overflow (100,000 examples)")
print("  ✓ PyTorch Documentation")

print("\n✅ Just Downloaded:")
for dataset, success in results.items():
    if success:
        print(f"  ✓ {dataset}")

total_datasets = 2 + successful

print(f"\n📊 TOTAL DATASETS: {total_datasets}")

if total_datasets >= 3:
    print("\n" + "="*70)
    print("✅ SUCCESS! You have enough datasets!")
    print("="*70)
    
    # Calculate total examples
    examples = {
        'Stack Overflow': 100000,
        'Code Alpaca': 20000,
        'Python Instructions': 18000,
        'MBPP': 1000
    }
    
    total_examples = 100000  # Stack Overflow
    for dataset, success in results.items():
        if success:
            total_examples += examples.get(dataset, 0)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total datasets: {total_datasets}")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Code examples: {total_examples - 100000:,}")
    
    print("\n🎯 What You Can Build:")
    print("  ✅ Semantic search (Stack Overflow)")
    print("  ✅ Documentation QA (PyTorch docs)")
    print("  ✅ Code generation (Code datasets)")
    print("  ✅ Instruction following (Alpaca datasets)")
    
    print("\n📝 Next Steps:")
    print("  1. Verify downloads: python inspect_datasets.py")
    print("  2. Start preprocessing (Week 1)")
    print("  3. Build baseline system (Week 2)")
    
    print("\n" + "="*70)
    print("🚀 READY TO START WEEK 2!")
    print("="*70)
    
else:
    print(f"\n⚠️  You have {total_datasets} datasets")
    print("Try downloading more options")

print()
