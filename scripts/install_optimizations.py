#!/usr/bin/env python3
"""
Install optimization packages for maximum performance
"""

import subprocess
import sys

def install_optimizations():
    """Install performance optimization packages"""
    
    print("🚀 Installing Performance Optimizations...")
    print("=" * 50)
    
    # Core optimizations
    packages = [
        # GPU acceleration (choose one based on your GPU)
        "cupy-cuda12x",  # For NVIDIA RTX 30/40 series
        # "cupy-cuda11x",  # For older NVIDIA GPUs
        
        # CPU optimizations
        "numba",         # JIT compilation
        "psutil",        # System monitoring
        
        # Parallel processing
        "joblib",        # Parallel computing
        
        # Fast linear algebra
        "openblas",      # Optimized BLAS
        
        # Memory optimization
        "memory-profiler",
    ]
    
    optional_packages = [
        "torch",         # PyTorch for GPU (alternative to CuPy)
        "intel-openmp",  # Intel optimizations
    ]
    
    print("Installing core optimizations...")
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} installation failed (may not be compatible)")
    
    print("\nInstalling optional optimizations...")
    for package in optional_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} installation failed (optional)")
    
    print("\n🎯 Optimization installation completed!")
    print("\nNext steps:")
    print("1. Run: python scripts/train_hybrid_model_gpu.py")
    print("2. Or run: python scripts/train_hybrid_model.py (original)")

if __name__ == "__main__":
    install_optimizations()