from decimal import Decimal
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import numba
from numba import jit
import timeit
import ctypes

lib = ctypes.cdll.LoadLibrary('../cimpl/cboard.so')
map_c = lib.applyTransform

num_pos = 64
batch_sizes = [2 ** 0, 2 ** 6, 2 ** 10, 2 ** 16]
torch_gpu_times = []
torch_cpu_times = []
np_times = []
np_acc_times = []
py_times = []
py_acc_times = []
for batch_size in batch_sizes:
    reps = int(2 ** 16 / batch_size)
    print("Batch size", batch_size)
    transform = np.arange(num_pos, dtype="int8") + 1
    transform[num_pos - 1] = 0
    base_batch = np.empty([batch_size, num_pos], dtype=np.int8)
    base_batch[:] = np.arange(num_pos, dtype=np.int8) * 2


    def map_np(base, t):
        start = time.time()
        for i in range(reps):
            lineVals = base[:, t]
        end = time.time() - start
        return lineVals, end


    @jit(nopython=True)
    def map_np_acc_(base, t):
        lineVals = base[:, t]
        return lineVals

    map_np_acc_(base_batch, transform)


    def map_np_acc(base, t):
        o = None
        start = time.time()
        for i in range(reps):
            o = map_np_acc_(base, t)
        end = time.time() - start
        return o, end


    @jit(nopython=True)
    def map_py_acc_(base, t):
        n = len(t)
        batch_out = []
        for row_num in range(len(base)):
            row = base[row_num]
            out = []
            for i in range(n):
                out.append(row[t[i]])
            batch_out.append(out)
        return batch_out


    map_py_acc_(base_batch, transform)


    def map_py_acc(base, t):
        o = None
        start = time.time()
        for i in range(reps):
            o = map_py_acc_(base, t)
        end = time.time() - start
        return o, end


    def map_py(base, t):
        start = time.time()
        n = len(t)
        batch_out = []
        for i in range(reps):
            for row in base:
                out = []
                for i in range(n):
                    out.append(row[t[i]])
                batch_out.append(out)
        end = time.time() - start
        return batch_out, end


    def map_torch(base, t):
        base = torch.tensor(base)
        t = torch.tensor(t, dtype=torch.long)
        base.to(device=torch.device('cuda'))
        t.to(device=torch.device('cuda'))
        start = time.time()
        for i in range(reps):
            lineVals = base[:, t]
        end = time.time() - start
        return lineVals, end


    def map_torch_cpu(base, t):
        base = torch.tensor(base)
        t = torch.tensor(t, dtype=torch.long)
        base.to(device=torch.device('cpu'))
        t.to(device=torch.device('cpu'))
        start = time.time()
        for i in range(reps):
            lineVals = base[:, t]
        end = time.time() - start
        return lineVals, end


    _, time_torch = map_torch(base_batch, transform)
    _, time_torch_cpu = map_torch_cpu(base_batch, transform)
    _, time_np = map_np(base_batch, transform)
    _, time_np_acc = map_np_acc(base_batch, transform)
    _, time_py = map_py(base_batch, transform)
    _, time_py_acc = map_py_acc(base_batch, transform)
    time_torch, time_torch_cpu, time_np, time_np_acc, time_py, time_py_acc = time_torch*1e3, time_torch_cpu*1e3, time_np*1e3, time_np_acc*1e3, time_py*1e3, time_py_acc*1e3
    print("Torch:", time_torch)
    print("Torch (CPU):", time_torch_cpu)
    print("Plain Python:", time_py)
    print("Numba Python:", time_py_acc)
    print("NP:", time_np)
    print("Numba NP:", time_np_acc)
    torch_gpu_times.append(time_torch)
    torch_cpu_times.append(time_torch_cpu)
    np_times.append(time_np)
    np_acc_times.append(time_np_acc)
    py_times.append(time_py)
    py_acc_times.append(time_py_acc)

ind = np.arange(len(batch_sizes))  # the x locations for the groups
width = 0.12  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width * 2.5, torch_gpu_times, width,
                label='Torch (GPU)')
rects2 = ax.bar(ind - width * 1.5, torch_cpu_times, width,
                label='Torch (CPU)')
rects3 = ax.bar(ind - width * .5, np_times, width,
                label='Numpy')
rects4 = ax.bar(ind + width * .5, np_acc_times, width,
                label='Numba + Numpy')
rects5 = ax.bar(ind + width * 1.5, py_times, width,
                label='Plain Python')
rects6 = ax.bar(ind + width * 2.5, py_acc_times, width,
                label='Numba Python')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Runtime (milliseconds)')
ax.set_title('Runtime by implementation and batch size for 65536 total "gather"s')
ax.set_xticks(ind)
ax.set_yticks([])
ax.set_xticklabels(batch_sizes)
ax.set_xlabel('Batch size')
ax.legend()


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('%o' % int(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(offset[xpos] * 3, 0),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=8)


autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")
autolabel(rects4, "center")
autolabel(rects5, "center")
autolabel(rects6, "center")

fig.tight_layout()

plt.show()
