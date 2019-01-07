//
// Created by Rel on 2019/1/6.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_SOLVER_H
#define SPARSE_KERNEL_TRIANGULAR_SOLVER_H

#include "sparse_matrix.h"


// Base class for triangular solvers
template <typename T>
class TriangularSolver {
public:
    // L: the lower triangular matrix
    // x: stores b at start and solution x at the end
    virtual int solve(SparseMatrix<T> &L,std::vector<T> &x) = 0;
};


#endif //SPARSE_KERNEL_TRIANGULAR_SOLVER_H
