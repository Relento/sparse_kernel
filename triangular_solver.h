//
// Created by Rel on 2019/1/6.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_SOLVER_H
#define SPARSE_KERNEL_TRIANGULAR_SOLVER_H

#include "sparse_matrix.h"


// Base class for sparse triangular solvers
template <typename T>
class TriangularSolver {
public:
    // L: the lower triangular matrix
    // NOTE: It is assumed that L's diagonal entries are all non-zeros
    // x: stores b at start and solution x at the end
    // verbose: if true, output helpful information during the computing process
    // Symbolically analyze the system, might do nothing in this function
    virtual int symAnalyze(SparseMatrix<T> &L,std::vector<T> &x,bool verbose = false) {return 0;}
    // Solve the system Lx = b, should be called **after** symAnalyze is called
    virtual int solve(SparseMatrix<T> &L,std::vector<T> &x,bool verbose = false) = 0;
};


#endif //SPARSE_KERNEL_TRIANGULAR_SOLVER_H
