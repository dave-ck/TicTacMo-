#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// pthreads vs bsp for C

int k = 0;
int n = 0;
int len = 0;
char **lines; //initialize to null and test for null when free-ing
int rowcount;
int colcount;


void initVars(int n_, int k_){
    k = k_;
    n = n_;
    len = pow(n, k); // len = n^k
    printf("Initialized Cimpl with n=%d, k=%d, len=%d\n", n, k, len);
}

void initLines(const void * indatav, int rowcount_in, int colcount_in) {
    // free each row separately, then top layer
    rowcount = rowcount_in;
    colcount = colcount_in;
    lines = malloc(rowcount*sizeof(char*));
    const char * indata = (char *) indatav;
    for (int row = 0; row < rowcount; ++row){
        lines[row] = malloc(colcount*sizeof(char*));
        for (int col = 0; col < colcount; ++col){
            int i = row * colcount + col;
            lines[row][col] = indata[i];
            printf("%d, %d: %d\n", row, col, indata[i]);
        }
    }

}

// for testing purposes only - verify that some input vector is a line
int isLine(const void * indatav, int rowcount, int colcount) {
    const char * indata = (char *) indatav;
    for (int row = 0; row < rowcount; ++row){
        int match = 1;
        int revMatch = 1; // check if reversed input is a match as well
        for (int col = 0; col < colcount; ++col){
            match = match && (lines[row][col] == indata[col]);
            revMatch = revMatch && (lines[row][col] == indata[colcount-1-col]);
        }
        if (match) return 1;
    }
    return 0;
}

int win(const void * indatav, char symbol){
//    printf("Checking for win with symbol: %d on board:\n", symbol);
    const char * board = (char *) indatav;
    for (int row = 0; row < rowcount; ++row){
        int line_win = 1;
        for(int col = 0; col < colcount; ++col){
            line_win = line_win && (board[lines[row][col]]==symbol);
        }
        if (line_win) return 1;
    }
    return 0;
}

int draw(const void * boardCells){
    const char * boardCells_ = (char *) boardCells;
    for (int i = 0; i < len; ++i) {
        if (boardCells_[i]==0) {
//            printf("Board cell %d was empty, held value %d\nSee board:\n", i, boardCells_[i]);
            printArr(boardCells);
            return 0; // if any cell is empty, return false
        }
//        else {
//            printf("Board cell %d was not empty, held value %d\n", i, boardCells_[i]);
//        }
    }
    return 1;
}

int printArr(const void * boardCells){
    const char * boardCells_ = (char *) boardCells;
    for (int i = 0; i < len; ++i) {
        printf("%d, ", boardCells_[i]);
    }
    printf("\n");
    return 1;
}

int printLines(){
    printf("printing %d lines\n", rowcount);
    for (int row = 0; row < rowcount; ++row){
        for(int col = 0; col < colcount; ++col){
            printf("line %d, index %d:%d\n", row, col, lines[row][col]);
        }
    }
    return 0;
}
