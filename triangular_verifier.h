//
// Created by Rel on 2019/1/7.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_VERIFIER_H
#define SPARSE_KERNEL_TRIANGULAR_VERIFIER_H

#include <iostream>
#include <string>
#include <vector>

#include "numerical_utils.h"
#include "sparse_matrix.h"
#include "triangular_solver.h"

template <typename T>
class TriangularVerifier{

    std::vector<SparseMatrix<T> *> Ls;
    std::vector<std::vector<T> *> bs;

    void loadTest(SparseMatrix<T> * L,std::vector<T>* b){
        Ls.push_back(L);
        bs.push_back(b);
    }

    bool verifySolver(TriangularSolver<T> s);
};

template<>
bool TriangularVerifier<double>::verifySolver(TriangularSolver<double> solver) {
    const double tol = 1e-5;
    bool flag = true;
    std::cout<<"residual norm tol:"<<tol<<std::endl;
    for (uint32_t i = 0; i<Ls.size();i++){
        auto &L = *(Ls[i]);
        auto &b = *(bs[i]);
        std::vector<double> x(b);
        solver.solve(L,x);
        double res = differenceNorm(L*x,b);
        std::cout<<"Test case #"<<i<<": col size: "<<Ls[i]->n<<" residual norm:"<<res<<std::endl;
        if (res>tol) flag = false;
    }
    return flag;
}

#endif //SPARSE_KERNEL_TRIANGULAR_VERIFIER_H
