import torch as T
import random

a = T.empty([3, 3 ** 2], dtype=T.int8)
for i in range(9):
    for j in range(3):
        a[j, i] = j * 9 + i
lines = T.tensor([[0, 1, 2],
                  [0, 3, 6],
                  [0, 4, 8],
                  [1, 4, 7],
                  [6, 4, 2],
                  [2, 5, 8],
                  [3, 4, 5],
                  [6, 7, 8]], dtype=T.int64)
print("With 0-26 fill:")
print("a:")
print(a)
print("a[:,lines]:")
print(a[:, lines])

a = T.randint(high=3, size=(4, 9)) - 1  # produce 4 random 3*3 boards
print("\n\nWith random in [-1,1] fill:")
print("a:")
print(a)
lineVals = a[:, lines]
print("lineVals:")
print(lineVals)

lineSums = T.sum(lineVals, axis=2)
print("line sums")
print(lineSums)
# lineSums contains an "n" (here 3) or a "-n" (-3) iff there is a line filled with -1's or with 1's
# take absolute value of lineSums
lineAbs = T.abs(lineSums)
print(lineAbs)
print((lineAbs == 3))
print((lineAbs == 3).any(axis=1))
wins = (lineAbs == 3).any(axis=1)
print(wins.nonzero())
