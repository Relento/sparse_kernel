//
// Created by Rel on 2019/1/7.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_PARA_H
#define SPARSE_KERNEL_TRIANGULAR_PARA_H

#include "triangular_solver.h"
#include <vector>
#include <deque>
#include <utility>
#include <omp.h>

template <typename T>
class TriangularPara : public TriangularSolver<T>{
public:
    bool para = true;
    int solve(SparseMatrix<T> &L, std::vector<T> &x);

    // Use BFS to calculate level set that could be parallelized TODO: this could be parallized as well
    std::vector<std::vector<uint32_t>> calLevelSet(SparseMatrix<T> &L);
};


template<typename T>
int TriangularPara<T>::solve(SparseMatrix<T> &L, std::vector<T> &x) {
    assert(L.m == L.n && L.m == x.size());

    std::vector<std::vector<uint32_t >> level_set;
    std::vector<uint32_t> nnz;

    auto t1 = high_resolution_clock::now();

    level_set = calLevelSet(L);

    auto t2 = high_resolution_clock::now();

//    std::cout<<"Non-zeros in b:"<<nnz.size()<<std::endl;
    std::cout<<"Size of level set:"<<level_set.size()<<std::endl;

//    uint32_t ct = 0;
//    for(auto &v:level_set){
//        std::cout<<v.size()<<" ";
//        ct+=v.size();
//    }
//    std::cout<<"nnz:"<<L.nnz<<" level ele size: "<<ct<<std::endl;

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
//    std::cout << "Calculate level_set Time:" << (double)duration / 1e6 <<std::endl;


    t1 = high_resolution_clock::now();
    for (auto &vec : level_set){

#pragma omp parallel for
        for (uint32_t i = 0;i<vec.size();i++){

            uint32_t j = vec[i];
            // Normalize diagonal entry
            x[j] /= L.values[L.outer_starts[j]];

            for(uint32_t p = L.outer_starts[j]+1 ; p < L.outer_starts[j+1] ; p++){
                x[L.inner_indices[p]] -= L.values[p] * x[j];
            }

        }
    }

    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>(t2-t1).count();

    std::cout << "Numerical Computation Time:" << (double)duration / 1e6 <<std::endl;
    return 0;
}

template<typename T>
std::vector<std::vector<uint32_t>> TriangularPara<T>::calLevelSet(SparseMatrix<T> &L) {

    std::vector<std::vector<uint32_t >> level_set;
    std::vector<uint32_t > indeg(L.m);

    for (uint32_t col = 0; col < L.n ; col++){
        for (uint32_t j = L.outer_starts[col]+1; j < L.outer_starts[col+1]; j++){
            indeg[L.inner_indices[j]]++;
        }
    }

    level_set.emplace_back(std::vector<uint32_t>());

    for(uint32_t i=0;i<L.m;i++){
        if(indeg[i] == 0){
            level_set[0].push_back(i);
        }
    }

    uint32_t cur_level = 0;

    while(!level_set[cur_level].empty()){
        cur_level++;
        level_set.emplace_back(std::vector<uint32_t>());
        for(uint32_t &ind:level_set[cur_level-1]){
            for (uint32_t j = L.outer_starts[ind] + 1; j < L.outer_starts[ind+1]; j++){
                uint32_t row = L.inner_indices[j];
                indeg[row]--;
                if(indeg[row] == 0){
                    level_set[cur_level].push_back(row);
                }
            }
        }
    }

    return level_set;
}

#endif //SPARSE_KERNEL_TRIANGULAR_PARA_H
