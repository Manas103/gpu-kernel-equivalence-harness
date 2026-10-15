#!/bin/bash
# Compiles matmul.cu into a standalone shared library for ctypes (see
# ../harness/cuda_kernel.py). Targets sm_89 (RTX 4070 SUPER / Ada). Rerun
# after editing matmul.cu; the .so is not committed (see .gitignore).
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler -fPIC -shared matmul.cu -o libmatmul.so
echo "built libmatmul.so"
