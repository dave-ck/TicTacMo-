from cimpl.cinit import generate_lines
import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./cboard.so')
draw = lib.draw
win = lib.win
init_lines = lib.initLines
init_vars = lib.initVars
c_print = lib.printArr
print_lines = lib.printLines

k = 2
n = 3

state_neutral = np.array([0, 0, 0,
                          0, 1, 0,
                          0, 0, -1], dtype='int8')

state_win = np.array([0, -1, 0,
                      1, -1, 1,
                      0, -1, 1], dtype='int8')

state_draw = np.array([1, -1, 1,
                       -1, -1, 1,
                       1, 1, -1], dtype='int8')

init_vars(n, k)
lines = generate_lines(n, k)
init_lines(ctypes.c_void_p(lines.ctypes.data), ctypes.c_int(lines.shape[0]), ctypes.c_int(lines.shape[1]))
print_lines()

print(state_neutral)
print("Draw:", draw(ctypes.c_void_p(state_neutral.ctypes.data)))
print("Win 1:", win(ctypes.c_void_p(state_neutral.ctypes.data), ctypes.c_int8(1)))
print("Win -1:", win(ctypes.c_void_p(state_neutral.ctypes.data), ctypes.c_int8(-1)))
c_print(ctypes.c_void_p(state_neutral.ctypes.data))

print(state_win)
print("Draw:", draw(ctypes.c_void_p(state_win.ctypes.data)))
print("Win 1:", win(ctypes.c_void_p(state_win.ctypes.data), ctypes.c_int8(1)))
print("Win -1:", win(ctypes.c_void_p(state_win.ctypes.data), ctypes.c_int8(-1)))
c_print(ctypes.c_void_p(state_win.ctypes.data))

print(state_draw)
print("Draw:", draw(ctypes.c_void_p(state_draw.ctypes.data)))
print("Win 1:", win(ctypes.c_void_p(state_draw.ctypes.data), ctypes.c_int8(1)))
print("Win -1:", win(ctypes.c_void_p(state_draw.ctypes.data), ctypes.c_int8(-1)))
c_print(ctypes.c_void_p(state_draw.ctypes.data))
