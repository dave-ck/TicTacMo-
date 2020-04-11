// compile with: gcc -fPIC -shared -o cboard.so board.c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// pthreads vs bsp for C

int k = 0;
int n = 0;
int numPos = 0;
char **lines; //initialize to null and test for null when free-ing
char **mappings;
int mappingCount;
int rowcount;
int colcount;


void initVars(int n_, int k_){
    k = k_;
    n = n_;
    numPos = pow(n, k); // numPos = n^k
    printf("Initialized Cimpl with n=%d, k=%d, numPos=%d\n", n, k, numPos);
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
        }
    }
}

void initMappings(const void * indatav, int mappingCount){
    mappings = malloc(mappingCount*sizeof(char*));
    const char * indata = (char *) indatav;
    for (int row = 0; row < mappingCount; ++row){
        lines[row] = malloc(k*sizeof(char*));
        for (int col = 0; col < k; ++col){
            int i = row * k + col;
            lines[row][col] = indata[i];
        }
    }
}

int arrLessThan(char * arr1, char * arr2, int numPos){
    for (int i = 0; i < numPos; ++i){
        if (arr1[i] < arr2[i]) return 1;
        if (arr1[i] >arr2[i]) return 0;
    }
    return 0;
}

void reduce(void * boardInCM){
    char * boardIn = (char *) boardInCM;
    char * current = malloc(numPos*sizeof(char*));
    char * best = malloc(numPos*sizeof(char*));
    memcpy((void *) boardIn, (void *) best, numPos);
    memcpy((void *) boardIn, (void *) current, numPos);
    for (int mapIndex = 0; mapIndex < mappingCount; ++mapIndex){
        for (int i=0; i < numPos; ++i){
            current[i] = boardIn[(mappings[mapIndex])[i]];
        }
        if (arrLessThan(current, best, numPos)){
            memcpy((void *) current, (void *) best, numPos);
        }
    }
    memcpy((void *) best, (void *) boardIn, numPos); // overwrite input array for output.
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
    for (int i = 0; i < numPos; ++i) {
        if (boardCells_[i]==0) {
            return 0;
        }
    }
    return 1;
}

int printArr(const void * boardCells){
    const char * boardCells_ = (char *) boardCells;
    for (int i = 0; i < numPos; ++i) {
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

void applyTransform(char * baseIn, char * transformIn, char * arrOutIn, int len){
//    printf("Casting...");
//    char * base = (char *) baseIn;
//    char * transform = (char *) transformIn;
//    char * arrOut = (char *) arrOutIn;
//    printf("Applying...");
    for (int i = 0; i < len; ++i){
        arrOutIn[i] = baseIn[(transformIn[i])];
    }
//    printf("Applied!");
}


