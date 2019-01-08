//
// Created by Rel on 2019/1/6.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_NAIVE_H
#define SPARSE_KERNEL_TRIANGULAR_NAIVE_H

#include "triangular_solver.h"
#include <omp.h>

template <typename T>
class TriangularNaive : public TriangularSolver<T>{
public:
    int solve(SparseMatrix<T> &L, std::vector<T> &x);
};

template<typename T>
int TriangularNaive<T>::solve(SparseMatrix<T> &L, std::vector<T> &x) {
    assert(L.m == L.n && L.m == x.size());

    for(uint32_t j = 0; j<L.n ;j++){
        // Normalize diagonal entry
        x[j] /= L.values[L.outer_starts[j]];

//#pragma omp parallel for
        for(uint32_t p = L.outer_starts[j]+1 ; p < L.outer_starts[j+1] ; p++){
            x[L.inner_indices[p]] -= L.values[p] * x[j];
        }
    }
    return 0;
}


#endif //SPARSE_KERNEL_TRIANGULAR_NAIVE_H
