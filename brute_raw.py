from win_check import win


# X goes first

def x_winnable(m, n, movesListO=[], movesListX=[], movesOpen=None, lastPlayed="O"):
    if movesOpen is None:  # can't use "if movesOpen" because the empty set needs to be admissible
        movesOpen = enumerate_moves(m, n)
    # check lastPlayed only, for performance
    if lastPlayed == "X" and win(movesListX, m):
        return True
    elif (lastPlayed == "O" and win(movesListO, m)) or len(movesOpen) == 0:  # once there are no moves left to play,
        # print("reached")
        return False
    winnable = (lastPlayed == "X")
    # print("\n")
    # print("movesListX at start", movesListX)
    # print("movesListO at start", movesListO)
    for move in movesOpen:
        mN = movesOpen[:]
        mX = movesListX[:]
        mO = movesListO[:]
        mN.remove(move)
        if lastPlayed == "X":
            mO.append(move)
            # print("reached if:")
            # print("mX", mX)
            # print("mO", mO)
            currentPlay = "O"
            winnable = winnable and x_winnable(m, n, mO, mX, mN, lastPlayed=currentPlay)
        else:
            mX.append(move)
            # print("reached else:")
            # print("mX", mX)
            # print("mO", mO)
            currentPlay = "X"
            winnable = winnable or x_winnable(m, n, mO, mX, mN, lastPlayed=currentPlay)
    return winnable


def enumerate_moves(n, k):
    # return the n**m possible moves
    # generate from value, converted to base m
    moves = []
    for val in range(n ** k):
        move = []
        # write, for i.e. 3 in base 2: 1,1,0,0...
        # first, write the correct representation in base n
        while val > 0:
            quo = val // n
            rem = val % n
            move.append(rem)
            val = quo
        # then, append zeroes to grow to size
        while len(move) < k:
            move.append(0)
        # no need to reverse, since we don't care about the order the moves are in
        moves.append(move)
    return moves


def batch():
    for m in range(1, 5):
        for n in range(1, 5):
            print("\n")
            if x_winnable(m, n):
                print("{}**{} is {}".format(m, n, "a win for X with best play."))
            else:
                print("{}**{} is {}".format(m, n, "a draw with best play."))


batch()
# print(x_winnable(2,2))

"""
# framework, useless in practice (checks wrong thing)
def brute_recurse_nomemo(m, n, movesListO, movesListX, movesN = None, lastPlayed="O"):
    if not movesN:
        movesN = enumerate_moves(m,n)
    # check lastPlayed only, for performance
    if lastPlayed == "X" and win(movesListX, m):
        return "X"
    if lastPlayed == "O" and win(movesListO, m):
        return "O"
    winners = set()
    for move in movesN:
        if lastPlayed == "X":
            movesListO.append(move)
            lastPlayed = "O"
        else:
            movesListX.append(move)
            lastPlayed="X"
        movesListN.remove(move)
        winners.add(brute_recurse_nomemo(m, n, movesListO, movesListX, lastPlayed))
        if "T" in winners or len(winners)>1:    # i.e. if there is a solution leading to a tie
            return "T"
    return str(winners[2])  # bodge but works
    """
