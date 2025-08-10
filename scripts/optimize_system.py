#!/usr/bin/env python3
"""
System optimization and performance tuning
"""

import os
import psutil
import platform
import subprocess

def optimize_system():
    """Optimize system settings for maximum performance"""
    
    print("🔧 System Optimization Report")
    print("=" * 50)
    
    # System info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # GPU detection
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
        else:
            print("GPU: No NVIDIA GPUs detected")
    except ImportError:
        print("GPU: Install GPUtil for GPU detection (pip install gputil)")
    
    # Environment optimizations
    print("\n🚀 Applying Optimizations...")
    
    # Set environment variables for performance
    optimizations = {
        'OMP_NUM_THREADS': str(psutil.cpu_count()),
        'MKL_NUM_THREADS': str(psutil.cpu_count()),
        'NUMEXPR_NUM_THREADS': str(psutil.cpu_count()),
        'OPENBLAS_NUM_THREADS': str(psutil.cpu_count()),
        'VECLIB_MAXIMUM_THREADS': str(psutil.cpu_count()),
        'NUMBA_NUM_THREADS': str(psutil.cpu_count()),
        
        # Memory optimizations
        'PYTHONHASHSEED': '0',
        'MALLOC_ARENA_MAX': '4',
        
        # NumPy optimizations
        'NPY_NUM_BUILD_JOBS': str(psutil.cpu_count()),
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        print(f"  ✅ {key} = {value}")
    
    # Performance recommendations
    print("\n💡 Performance Recommendations:")
    
    if memory.available < 8 * (1024**3):  # Less than 8GB
        print("  ⚠️  Consider closing other applications to free up RAM")
    
    if psutil.cpu_count() >= 8:
        print("  ✅ Multi-core CPU detected - parallel processing will be effective")
    
    # Check for SSD
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        print(f"  💾 Disk space: {disk_usage.free / (1024**3):.1f} GB free")
    except:
        pass
    
    print("\n🎯 System optimized for maximum performance!")
    
    return optimizations

if __name__ == "__main__":
    optimize_system()