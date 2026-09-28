"""ctypes binding for the one-kernel CUDA global-feature selector."""
from __future__ import annotations
import ctypes
from pathlib import Path
import torch

_LIB_PATH = Path(__file__).with_name("libcoverage_greedy_cuda.so")
_LIB = None

def _library():
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.is_file():
            raise RuntimeError(f"missing CUDA coverage library: {_LIB_PATH}")
        lib = ctypes.CDLL(str(_LIB_PATH))
        f = lib.vispruner_global_greedy_cuda
        void = ctypes.c_void_p
        f.argtypes = [void, void, void, void, ctypes.c_int, ctypes.c_int, void, void, void, void, void]
        f.restype = ctypes.c_int
        _LIB = lib
    return _LIB

def global_feature_greedy_cuda(rel, glob, user_similarity, key_similarity, keep_n: int) -> torch.Tensor:
    tensors = (rel, glob, user_similarity, key_similarity)
    if any(not torch.is_tensor(x) or not x.is_cuda or x.dtype != torch.float32 for x in tensors):
        raise ValueError("CUDA coverage inputs must be CUDA float32 tensors")
    rel, glob, user_similarity, key_similarity = (x.contiguous() for x in tensors)
    n = int(rel.numel()); keep = int(keep_n)
    if glob.shape != (n,) or user_similarity.shape != (n, n) or key_similarity.shape != (n, n):
        raise ValueError("CUDA coverage shape mismatch")
    if any(x.device != rel.device for x in (glob, user_similarity, key_similarity)):
        raise ValueError("CUDA coverage device mismatch")
    available = torch.empty(n, device=rel.device, dtype=torch.uint8)
    user_res = torch.empty(n, device=rel.device, dtype=torch.float32)
    key_res = torch.empty(n, device=rel.device, dtype=torch.float32)
    out = torch.empty(keep, device=rel.device, dtype=torch.int64)
    stream = torch.cuda.current_stream(rel.device)
    ptr = ctypes.c_void_p
    code = _library().vispruner_global_greedy_cuda(
        ptr(rel.data_ptr()), ptr(glob.data_ptr()), ptr(user_similarity.data_ptr()), ptr(key_similarity.data_ptr()),
        n, keep, ptr(available.data_ptr()), ptr(user_res.data_ptr()), ptr(key_res.data_ptr()),
        ptr(out.data_ptr()), ptr(stream.cuda_stream),
    )
    if code != 0:
        raise RuntimeError(f"CUDA coverage greedy launch failed with code {code}")
    return out
