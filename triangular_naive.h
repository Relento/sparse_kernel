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
    int solve(SparseMatrix<T> &L, std::vector<T> &x, bool verbose = false);
};

template<typename T>
int TriangularNaive<T>::solve(SparseMatrix<T> &L, std::vector<T> &x,bool verbose) {
    assert(L.m == L.n && L.m == x.size());

    // Convert vector to arrays to speed up
    T *Lv = &(L.values[0]);
    T *xv = &(x[0]);
    uint32_t  *Lo = &(L.outer_starts[0]);
    uint32_t  *Li = &(L.inner_indices[0]);

    for(uint32_t j = 0; j<L.n ;j++){
        // Normalize diagonal entry
        xv[j] /= Lv[Lo[j]];

//omp_set_num_threads(1);
//#pragma omp parallel for
        for(uint32_t p = Lo[j]+1 ; p < Lo[j+1] ; p++){
//#pragma omp atomic update
            xv[Li[p]] -= Lv[p] * xv[j];
        }

    }
    return 0;
}


#endif //SPARSE_KERNEL_TRIANGULAR_NAIVE_H
