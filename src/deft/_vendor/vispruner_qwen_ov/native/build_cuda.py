#!/usr/bin/env python3
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
subprocess.run([
    "/usr/local/cuda/bin/nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC",
    "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
    str(root/"coverage_greedy_cuda.cu"), "-o", str(root/"libcoverage_greedy_cuda.so")
], check=True)
print(root/"libcoverage_greedy_cuda.so")
