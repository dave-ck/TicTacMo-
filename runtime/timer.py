import torch
import time
import numpy as np
import numba
from numba import jit
import timeit
import ctypes

lib = ctypes.cdll.LoadLibrary('../cimpl/cboard.so')
map_c = lib.applyTransform

num_pos = 127  # 5**5
reps = 1e5
transform = np.arange(num_pos, dtype="int8") + 1
transform[num_pos - 1] = 0
base = np.arange(num_pos, dtype="int8") * 2
out = np.zeros(num_pos, dtype="int8")

print("empty", out)
map_c(ctypes.c_void_p(base.ctypes.data), ctypes.c_void_p(transform.ctypes.data), ctypes.c_void_p(out.ctypes.data),
      ctypes.c_int(num_pos))
print("filled", out)


def map_np(b, t):
    return b[t]


@jit(nopython=True)
def map_np_acc(b, t):
    return b[t]


def map_torch(b, t):
    return b[t]


def map_py(b, t, n):
    out = []
    for i in range(n):
        out.append(b[t[i]])
    return out


@jit(nopython=True)
def map_py_acc(b, t, n):
    out = []
    for i in range(n):
        out.append(b[t[i]])
    return out


print(map_np(base, transform))
print(map_np_acc(base, transform))
print(map_py(base, transform, num_pos)[:10])
print(map_py_acc(base, transform, num_pos)[:10])

start = time.time()
for _ in range(int(reps)):
    map_np(base, transform)
print("Map_np", time.time() - start)

start = time.time()
for _ in range(int(reps)):
    map_np_acc(base, transform)
print("Map_np_acc", time.time() - start)

base_t = torch.tensor(base, dtype=torch.long)
transform_t = torch.tensor(transform, dtype=torch.long)
start = time.time()
for _ in range(int(reps)):
    map_torch(base_t, transform_t)
print("Map_torch_cpu", time.time() - start)

base_t.cuda()
transform_t.cuda()
start = time.time()
for _ in range(int(reps)):
    map_torch(base_t, transform_t)
print("Map_torch_cuda", time.time() - start)

"""
start = time.time()
for _ in range(in
(reps)):
    map_py(base, transform, num_pos)
print("Map_py", time.time()-start)
"""

start = time.time()
for _ in range(int(reps)):
    map_py_acc(base, transform, num_pos)
print("Map_py_acc", time.time() - start)

start = time.time()
for _ in range(int(reps)):
    map_c(ctypes.c_void_p(base.ctypes.data), ctypes.c_void_p(transform.ctypes.data), ctypes.c_void_p(out.ctypes.data),
          ctypes.c_int(num_pos))
print("Map_c", time.time() - start)

b = ctypes.c_void_p(base.ctypes.data)
t = ctypes.c_void_p(transform.ctypes.data)
o = ctypes.c_void_p(out.ctypes.data)
n = ctypes.c_int(num_pos)
start = time.time()
for _ in range(int(reps)):
    map_c(b, t, o, n)
print("Map_c_precast", time.time() - start)
