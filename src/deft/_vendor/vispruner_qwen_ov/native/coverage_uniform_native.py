"""ctypes binding for the single-space uniform-start greedy selector."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).with_name("libcoverage_uniform_start.so")
_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.is_file():
            raise RuntimeError(f"missing native uniform-start library: {_LIB_PATH}")
        lib = ctypes.CDLL(str(_LIB_PATH))
        function = lib.vispruner_uniform_start_greedy
        float_ptr = ctypes.POINTER(ctypes.c_float)
        int64_ptr = ctypes.POINTER(ctypes.c_int64)
        function.argtypes = [
            float_ptr,
            float_ptr,
            float_ptr,
            ctypes.c_int,
            ctypes.c_int,
            int64_ptr,
        ]
        function.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def global_feature_uniform_start_greedy(
    relevance,
    global_importance,
    similarity,
    keep_n: int,
) -> list[int]:
    arrays = [
        np.ascontiguousarray(value, dtype=np.float32)
        for value in (relevance, global_importance, similarity)
    ]
    count = int(arrays[0].shape[0])
    keep = int(keep_n)
    if arrays[1].shape != (count,) or arrays[2].shape != (count, count):
        raise ValueError("native uniform-start coverage shape mismatch")
    if keep <= 0 or keep > count:
        raise ValueError("native uniform-start keep count is invalid")
    output = np.empty(keep, dtype=np.int64)
    float_ptr = ctypes.POINTER(ctypes.c_float)
    code = _library().vispruner_uniform_start_greedy(
        arrays[0].ctypes.data_as(float_ptr),
        arrays[1].ctypes.data_as(float_ptr),
        arrays[2].ctypes.data_as(float_ptr),
        count,
        keep,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if code != 0:
        raise RuntimeError(f"native uniform-start coverage failed with code {code}")
    return [int(value) for value in output]
