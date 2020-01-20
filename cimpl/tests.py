import ctypes
import numpy

lib = ctypes.cdll.LoadLibrary('./cboard.so')
draw = lib.draw
init_vars = lib.init_vars
k = 3
n = 4
ones = numpy.ones((n ** k), dtype=numpy.uint)
zeros = numpy.zeros((n ** k), dtype=numpy.int)
mix = zeros.copy()
mix[5:8] = 1
mix[20:23] = 1
mix = mix * 212



ones[20:] = 3
ones = ones * 2
init_vars(k, n)


print(ones)
print(draw(ctypes.c_void_p(ones.ctypes.data)))  # should be true



print(zeros)
print(draw(ctypes.c_void_p(zeros.ctypes.data)))  # should be false

print(mix)
print(draw(ctypes.c_void_p(mix.ctypes.data)))  # should be true

