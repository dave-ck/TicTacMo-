from matplotlib import pyplot as plt
import torch
import time
import numpy as np
import numba
from numba import jit
import timeit
import ctypes

lib = ctypes.cdll.LoadLibrary('../cimpl/cboard.so')
map_c = lib.applyTransform

torch_gpu_times = []
torch_cpu_times = []
np_times = []
np_acc_times = []
py_times = []
py_acc_times = []
c_times = []
precast_c_times = []
pos_nums = [9, 16, 27, 64, 128]
for num_pos in pos_nums:
    reps = 2**16
    transform = np.arange(num_pos, dtype="int8") + 1
    transform[num_pos - 1] = 0
    base = np.arange(num_pos, dtype="int8") * 2
    out = np.zeros(num_pos, dtype="int8")

    print("empty", out)
    map_c(ctypes.c_void_p(base.ctypes.data), ctypes.c_void_p(transform.ctypes.data), ctypes.c_void_p(out.ctypes.data),
          ctypes.c_int(num_pos))
    print("filled", out)


    def map_np(b, t):
        start = time.time()
        for i in range(reps):
            o = b[t]
        end = time.time() - start
        return o, end

    @jit(nopython=True)
    def map_np_acc_(b, t):
        return b[t]


    map_np_acc_(base, transform)


    def map_np_acc(b, t, reps):
        start = time.time()
        for i in range(reps):
            o = map_np_acc_(b, t)
        end = time.time() - start
        return o, end


    def map_torch(b, t):
        start = time.time()
        o = b[t]
        end = time.time() - start
        return


    def map_torch_gpu(b, t):
        b = torch.tensor(b)
        t = torch.tensor(t, dtype=torch.long)
        b.to(device=torch.device('cuda'))
        t.to(device=torch.device('cuda'))
        lineVals = None
        start = time.time()
        for i in range(reps):
            lineVals = b[t]
        end = time.time() - start
        return lineVals, end

    def map_torch_cpu(b, t):
        b = torch.tensor(b)
        t = torch.tensor(t, dtype=torch.long)
        b.to(device=torch.device('cpu'))
        t.to(device=torch.device('cpu'))
        lineVals = None
        start = time.time()
        for i in range(reps):
            lineVals = b[t]
        end = time.time() - start
        return lineVals, end

    def map_py(b, t):
        out = []
        n = len(t)
        start = time.time()
        for j in range(reps):
            for i in range(n):
                out.append(b[t[i]])
        end = time.time()-start
        return out, end




    @jit(nopython=True)
    def map_py_acc_(b, t):
        out = []
        n = len(t)
        for i in range(n):
            out.append(b[t[i]])
        return out


    map_py_acc_(base, transform)


    def map_py_acc(b, t, reps):
        start = time.time()
        for i in range(reps):
            o = map_py_acc_(b, t)
        end = time.time() - start
        return o, end


    def map_c_plain(b, t, o):
        num_pos = len(t)
        start = time.time()
        for _ in range(int(reps)):
            map_c(ctypes.c_void_p(b.ctypes.data), ctypes.c_void_p(t.ctypes.data), ctypes.c_void_p(o.ctypes.data),
                  ctypes.c_int(num_pos))
        end = time.time() - start
        return o, end


    def map_c_precast(b, t, o):
        num_pos = len(t)
        b = ctypes.c_void_p(b.ctypes.data)
        t = ctypes.c_void_p(t.ctypes.data)
        o = ctypes.c_void_p(o.ctypes.data)
        n = ctypes.c_int(num_pos)
        start = time.time()
        for _ in range(int(reps)):
            map_c(b, t, o, n)
        end = time.time() - start
        return o, end

    _, time_torch = map_torch_gpu(base, transform)
    _, time_torch_cpu = map_torch_cpu(base, transform)
    _, time_np = map_np(base, transform)
    _, time_np_acc = map_np_acc(base, transform, reps)
    _, time_py = map_py(base, transform)
    _, time_py_acc = map_py_acc(base, transform, reps)
    _, time_c = map_c_plain(base, transform, out)
    _, time_c_precast = map_c_precast(base, transform, out)
    print("Torch:", time_torch)
    print("Torch (CPU):", time_torch_cpu)
    print("Plain Python:", time_py)
    print("Numba Python:", time_py_acc)
    print("NP:", time_np)
    print("Numba NP:", time_np_acc)
    print("Cast C:", time_c)
    print("Precast C:", time_c_precast)
    torch_gpu_times.append(time_torch)
    torch_cpu_times.append(time_torch_cpu)
    np_times.append(time_np)
    np_acc_times.append(time_np_acc)
    py_times.append(time_py)
    py_acc_times.append(time_py_acc)
    c_times.append(time_c)
    precast_c_times.append(time_c_precast)

ind = np.arange(len(pos_nums))  # the x locations for the groups
width = 0.12  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width * 3.5, torch_gpu_times, width,
                label='GPU')
rects2 = ax.bar(ind - width * 2.5, torch_cpu_times, width,
                label='CPU')
rects3 = ax.bar(ind - width * 1.5, np_times, width,
                label='NP')
rects4 = ax.bar(ind - width * .5, np_acc_times, width,
                label='Numba NP')
rects5 = ax.bar(ind + width * .5, py_times, width,
                label='Plain Python')
rects6 = ax.bar(ind + width * 1.5, py_acc_times, width,
                label='Numba Python')
rects7 = ax.bar(ind + width * 2.5, c_times, width,
                label='C with cast')
rects8 = ax.bar(ind + width * 3.5, precast_c_times, width,
                label='C precast')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Runtime (milliseconds)')
ax.set_title('Runtime by implementation for performing "gather" 2**16 times')
ax.set_xticks(ind)
ax.set_xticklabels(pos_nums)
ax.set_xlabel('Array size')
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
        ax.annotate('%o' % int(height * 1000),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(offset[xpos] * 3, 0),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=5)


autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")
autolabel(rects4, "center")
autolabel(rects5, "center")
autolabel(rects6, "center")
autolabel(rects7, "center")
autolabel(rects8, "center")

fig.tight_layout()

plt.show()

