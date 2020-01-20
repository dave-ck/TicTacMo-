#include <stdio.h>
#include <math.h>

int k = 0;
int n = 0;
int len = 0;

void init_vars(int k_, int n_){
    k = k_;
    n = n_;
    len = pow(n, k); // len = n^k
    printf("Initialized Cimpl with n=%d, k=%d, len=%d\n", n, k, len);
}

int draw(const void * boardCells){
    const int * boardCells_ = (int *) boardCells;
    for (int i = 0; i < len; ++i) {
        // printf("%d:%d\n", i, boardCells_[i]);
        if (boardCells_[i]==0) return 0; // if any cell is empty, return false
    }
    return 1;
}

//bool win(const void * board, int len, int symbol) {
//    // return true if symbol has a winning set in the input board.
//    const int * board = (int *) indatav;
//    for (int i = 0; i < len; i++) {
//        out &= (board[i] > k);
//    }
//
//
//	}
//
//int read(const void * board, const void * vector, int len){
//    // return the symbol held at coordinates given by the vector
//
//}
//
//
//
//def win_imperative(moves_list, n):
//    print("Trying {}, {}".format(moves_list, n))
//    if not moves_list:
//        return False
//    moves_list = sorted(moves_list)
//    # choose the starting vector.
//    for vector0 in moves_list:
//        # Any winning set will necessarily include at least one vector containing at least one zero.
//        # Assume wlog that this is vector1.
//        if 0 not in vector0:
//            continue
//        # choose a second vector different from the first
//        for vector1 in moves_list:
//            if vector0 == vector1:
//                continue
//            # print("Trying pair", vector0, ",", vector1)
//            # calculate the "gradient"
//            grads = []
//            for i in range(len(vector0)):  # same len as vector1 - may need error checking
//                grads.append(vector1[i] - vector0[i])
//            # if the 2 vectors are NOT adjacent, break:
//            adjacent = True
//            for grad in grads:
//                adjacent = adjacent and grad in [-1, 0, 1]
//            if not adjacent:
//                continue
//            # Compute each v_i which would be in the solution V
//            v_i = vector0[:]
//            i = 0
//            while v_i in moves_list:
//                i += 1
//                # compute next v_(i+1) by summing v_i and grads pairwise
//                for j in range(len(v_i)):  # probably good to just define vector length n as a parameter at func start
//                    v_i[j] += grads[j]  # increment current grid cell by grads
//            if i == n:  # if the while-loop's condition evaluated to "True" n times
//                return True
//    return False
//*/