board_1 = {"X": [[1, 1]], "O": []}
#   0 1 2
# 0       0
# 1   X   1
# 2       2
#   0 1 2
reduced_board_1 = {"X": [[1, 1]], "O": []}
#   0 1 2
# 0       0
# 1   X   1
# 2       2
#   0 1 2


board_2 = {"X": [[1, 1]], "O": [[0, 2]]}
#   0 1 2
# 0     O 0
# 1   X   1
# 2       2
#   0 1 2
reduced_board_2 = {"X": [[1, 1]], "O": [[0, 0]]}
#   0 1 2
# 0 O     0
# 1   X   1
# 2       2
#   0 1 2


board_3 = {"X": [[1, 1], [1, 2]], "O": [[0, 2]]}
#   0 1 2
# 0     O 0
# 1   X X 1
# 2       2
#   0 1 2
reduced_board_3 = {"X": [[1, 1], [0, 1]], "O": [[0, 0]]}
#   0 1 2
# 0 O X   0
# 1   X   1
# 2       2
#   0 1 2


board_4 = {"X": [[1, 1], [1, 2]], "O": [[0, 2], [1, 0]]}
#   0 1 2
# 0     O 0
# 1 O X X 1
# 2       2
#   0 1 2
reduced_board_4 = {"X": [[1, 1], [0, 1]], "O": [[0, 0], [2, 1]]}
#   0 1 2
# 0 O X   0
# 1   X   1
# 2   O   2
#   0 1 2


board_5 = {"X": [[1, 1], [1, 2], [0, 0]], "O": [[0, 2], [1, 0]]}
#   0 1 2
# 0 X   O 0
# 1 O X X 1
# 2       2
#   0 1 2
reduced_board_5 = {"X": [[1, 1], [1, 2], [0, 0]], "O": [[0, 2], [1, 0]]}
#   0 1 2
# 0 X   O 0
# 1 O X X 1
# 2       2
#   0 1 2

def board_reduce(board, n, k):
    # note: functions max and min work on iterables of iterables
    # >>> max([[0, 0], [0, 1], [1, 0, 0], [1, 1]])
    # [1, 1]
    X = board["X"]
    O = board["O"]
    # store the current transformation as a tuple/array of size K
    invert_constraint = [None for _ in range(k)]
    swap_with_constraint = [[i for i in range(j, k)] for j in range(k)]
    # find the "most extreme" move in X, wrt constraints:
    fully_constrained = False
    while not fully_constrained: # todo: impl fully_constrained flag
        constraint_target = 0
        target_val = 0  # impose total order; target with higher val produces a "lower" move once transformed
        for i_x in range(len(X)): # for each vector x in the set of moves X
            x = X[i_x]
            opt_x = x.copy() # apply transformations to opt_x (for comparison)
            # for each unconstrained value in the vector x
            for i_val in range(len(x)):
                val = x[i_val]
                if invert_constraint[i_val] is None:
                    # score with best swapperino
                    pass
    raise NotImplementedError

def naive_reduce(board, n, k):
    # produce *every* equivalent board, then return the least of these
    # encode transforms as a list of tuples [(address_swap, invert_val)]
    # address_swap: the address (index) to send the current index-held value to
    # invert_val: boolean indicating whether or not the value needs to be inverted (prior to swap)
    # i.e. to transform [a,b,c] to [n-c-1, b, a], would have transform [(3, True), (2, False), (1, False)]
    raise NotImplementedError

# need only enumerate such moves once
def enum_transforms(k):
    swaps = [[i for i in range(k)]]


    return swaps

print(enum_transforms(6))





def legal_swaps(swap_with, k, n):
    build_list = [swap_with]
    for i in range(k):
        if swap_with[i] is None:
            for elem in build_list.copy():  # Cimpl: can be done in-place
                for swap_val in range(i, n):
                    elem_v = elem.copy()
                    elem_v[i] = swap_val
                    build_list.append(elem_v)
                build_list.remove(elem)
    return build_list


def legal_inversions(invert, k):
    inversions_return = [invert]
    for i in range(k):
        if invert[i] is None:
            additions = []
            for invert_1 in inversions_return:
                invert_2 = invert_1.copy()
                invert_1[i] = True
                invert_2[i] = False
                additions.append(invert_2)
            inversions_return.extend(additions)
    return inversions_return


def apply_tranform(invert, swap_with, move, n, k):
    if None in invert or None in swap_with:
        raise ValueError("Attempted to compute move with a None value in one of the constraints.")
    for i in range(k):
        if invert[i]:
            move[i] = (n - 1) - 1 * move[i]
    print("Inversion applied:", move)
    for i in range(k):
        if swap_with[i]:  # swap_with[i] is either a non-zero positive int, or False
            j = swap_with[i]
            move[i], move[j] = move[j], move[i]
            print("Swap {} for {} applied:".format(i, j), move)
    return move

#
# original_move = [0, 1, 2, 3, 4]
# n = 5
# k = 5
# invert = [True, True, False, True, False]
# swap_with = [3, False, False, False, False]
# result_move = [1, 3, 2, 4, 4]
# print(apply_tranform(invert, swap_with, original_move, n, k))
#
# invert = [False, None, None, False]
# print(legal_inversions(invert, 4))
#
# swap_with = [[0], [1, 2], [1, 2]]
