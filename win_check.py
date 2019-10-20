"""
We rely on the property, wlog, that any win will have ONE "minimal" vector v_0 -
That is, for every solution V, there is some v_0 which contains at least one 0 (say at index i), such that:
    no other v_j contains a 0 at index i
    every v_k in V contains k at index i (we'll call index i our INCREMENTATION INDEX (name is a work in progress, we'll find something catchier))
"""


def win(movesList, m):
    movesList = sorted(movesList)
    # choose the starting vector.
    for vector0 in movesList:
        # Any winning set will necessarily include at least one vector containing at least one zero. Make this vector 1 WLOG.
        if 0 not in vector0:
            continue
        # choose a second vector different from the first
        for vector1 in movesList:
            if vector0 == vector1:
                continue
            # print("Trying pair", vector0, ",", vector1)
            # calculate the "gradient"
            grads = [x1 - x0 for (x0, x1) in zip(vector0, vector1)]
            # print("Gradient vector is:", grads)
            # if the 2 vectors are NOT adjacent, break:
            if not all(map(lambda x: x in [-1, 0, 1], grads)):
                # print("Not adjacent")
                continue
            pairing = True  # Assume vector1, vector2 belong to some solution V
            # Compute each v_i which would be in the solution V
            v_i = vector1
            for i in range(2, m):  # start at 2, no need to check v_0 and v_1
                v_i = list(map(sum, zip(v_i, grads)))
                if v_i not in movesList:
                    # print(v_i, "was absent from the movesList")
                    pairing = False
                    break  # no need to continue with loop (note for refactor: can use while-loop, but will probably look clunkier)
                # print(v_i, "in solution")
            if pairing:
                return True
    return False


##############################################################################
######################         TEST CODE BELOW             ###################
##############################################################################


"""    
winTest0 = (
    [[1,0], #vanilla win, vertical/horizontal
     [0,0],
     [2,0]],
    3)


winTest1 = (
    [[1,1], #vanilla win, diagonal
     [0,0],
     [2,2]],
    3)

winTest2 = (
    [[1,1],
     [0,0],
     [2,1], #spoiler
     [2,2]],
    3)

winTestHighDim = (
    [[1,1,2,2,1],
     [0,0,1,1,1],
     [2,1,1,1,0],
     [2,2,2,2,0],
     [2,0,0,0,0]],
    3)

winTestBigBoard = (
    [[i,1] for i in range(20)]+[[i%7+1, i+2] for i in range(23)]+[[i,1] for i in range(21,40)],
    40)

loseTest = (
    [[1,2],
     [0,0],
     [1,1], 
     [2,0]],
    3)



print("Win tests")
print("\n")
print(win(*winTest0))
print("\n")
print(win(*winTest1))
print("\n")
print(win(*winTest2))
print("\n")
print("\n")
print("Lose tests")
print("\n")
print(win(*loseTest))
print("\n")
print("\n")
print("5D - win")
print("\n")
print(win(*winTestHighDim))

print("\n")
print("\n")
print("High Size, 2D - win")
print("\n")
print(win(*winTestBigBoard))
"""
