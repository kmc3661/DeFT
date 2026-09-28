"""Thin ctypes binding for the fixed global-feature greedy selector."""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np

_LIB_PATH = Path(__file__).with_name("libcoverage_greedy.so")
_LIB = None

def _library():
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.is_file():
            raise RuntimeError(f"missing native coverage library: {_LIB_PATH}")
        lib = ctypes.CDLL(str(_LIB_PATH))
        f = lib.vispruner_global_greedy
        float_ptr = ctypes.POINTER(ctypes.c_float)
        int64_ptr = ctypes.POINTER(ctypes.c_int64)
        f.argtypes = [float_ptr, float_ptr, float_ptr, float_ptr, ctypes.c_int, ctypes.c_int, int64_ptr]
        f.restype = ctypes.c_int
        truncated = lib.vispruner_global_truncated_greedy
        truncated.argtypes = [float_ptr, float_ptr, float_ptr, float_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, int64_ptr]
        truncated.restype = ctypes.c_int
        _LIB = lib
    return _LIB

def global_feature_greedy(rel, glob, user_similarity, key_similarity, keep_n: int) -> list[int]:
    arrays = [np.ascontiguousarray(x, dtype=np.float32) for x in (rel, glob, user_similarity, key_similarity)]
    n = int(arrays[0].shape[0]); keep = int(keep_n)
    if arrays[1].shape != (n,) or arrays[2].shape != (n,n) or arrays[3].shape != (n,n):
        raise ValueError("native coverage shape mismatch")
    out = np.empty(keep, dtype=np.int64)
    fp = ctypes.POINTER(ctypes.c_float)
    code = _library().vispruner_global_greedy(
        arrays[0].ctypes.data_as(fp), arrays[1].ctypes.data_as(fp),
        arrays[2].ctypes.data_as(fp), arrays[3].ctypes.data_as(fp),
        n, keep, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if code != 0:
        raise RuntimeError(f"native coverage greedy failed with code {code}")
    return [int(x) for x in out]

def global_feature_truncated_greedy(rel, glob, user_similarity, key_similarity, keep_n: int, steps: int) -> list[int]:
    arrays = [np.ascontiguousarray(x, dtype=np.float32) for x in (rel, glob, user_similarity, key_similarity)]
    n = int(arrays[0].shape[0]); keep = int(keep_n); steps = max(1, min(int(steps), keep))
    if arrays[1].shape != (n,) or arrays[2].shape != (n,n) or arrays[3].shape != (n,n):
        raise ValueError("native truncated coverage shape mismatch")
    out = np.empty(keep, dtype=np.int64); fp = ctypes.POINTER(ctypes.c_float)
    code = _library().vispruner_global_truncated_greedy(
        arrays[0].ctypes.data_as(fp), arrays[1].ctypes.data_as(fp),
        arrays[2].ctypes.data_as(fp), arrays[3].ctypes.data_as(fp),
        n, keep, steps, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if code != 0:
        raise RuntimeError(f"native truncated coverage greedy failed with code {code}")
    return [int(x) for x in out]

