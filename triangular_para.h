//
// Created by Rel on 2019/1/7.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_PARA_H
#define SPARSE_KERNEL_TRIANGULAR_PARA_H

#include "triangular_solver.h"

template <typename T>
class TriangularPara : public TriangularSolver<T>{
public:
    int solve(SparseMatrix<T> &L, std::vector<T> &x);
};

template<typename T>
int TriangularPara<T>::solve(SparseMatrix<T> &L, std::vector<T> &x) {
    if(L.m != L.n || L.m != x.size())
        return -1;
    for(uint32_t j = 0; j<L.n ;j++){
        // Normalize diagonal entry
        x[j] /= L.values[L.outer_starts[j]];

        for(uint32_t p = L.outer_starts[j]+1 ; p < L.outer_starts[j+1] ; p++){
            x[L.inner_indices[p]] -= L.values[p] * x[j];
        }
    }
    return 0;
}

#endif //SPARSE_KERNEL_TRIANGULAR_PARA_H