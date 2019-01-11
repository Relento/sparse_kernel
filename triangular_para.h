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
#include <unordered_map>
#include <cstdint>
#include <stdint-gcc.h>

template <typename T>
class TriangularPara : public TriangularSolver<T>{
public:
    uint32_t thread_num = 8;
    std::vector<std::vector<uint32_t >> level_set;
    enum class LevelSetType {Normal,Average,Prune} cal_level = LevelSetType::Normal;
    int symAnalyze(SparseMatrix<T> &L, std::vector<T> &x,bool verbose = false);
    int solve(SparseMatrix<T> &L, std::vector<T> &x,bool verbose = false);

    // Use BFS to calculate level set that could be parallelized TODO: this could be parallized as well
    std::vector<std::vector<uint32_t>> calLevelSet(SparseMatrix<T> &L);
    std::vector<std::vector<uint32_t>> calLevelSetPrune(SparseMatrix<T> &L,std::vector<uint32_t> b_nnz);
    std::vector<std::vector<uint32_t>> calLevelSetAverage(SparseMatrix<T> &L);
    void dfs(uint32_t node,SparseMatrix<T> &L, std::vector<bool> &visit, std::vector<uint32_t> &reach_set);
};


template<typename T>
int TriangularPara<T>::solve(SparseMatrix<T> &L, std::vector<T> &x,bool verbose) {



    for (auto &vec : level_set){

#pragma omp parallel for
        for (uint32_t i = 0;i<vec.size();i++){

            uint32_t j = vec[i];
            // Normalize diagonal entry
            x[j] /= L.values[L.outer_starts[j]];

//#pragma omp parallel for
            for(uint32_t p = L.outer_starts[j]+1 ; p < L.outer_starts[j+1] ; p++){
#pragma omp atomic update // assure consistency
                x[L.inner_indices[p]] -= L.values[p] * x[j];
            }

        }
    }

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

template<typename T>
std::vector<std::vector<uint32_t>> TriangularPara<T>::calLevelSetAverage(SparseMatrix<T> &L) {
    std::vector<std::vector<uint32_t >> level_set(thread_num);
    uint32_t average = L.n/thread_num;
    level_set.emplace_back(std::vector<uint32_t>());
    for(uint32_t col = 0; col < L.n ; col++){
        uint32_t ind = col/average;
        if(ind>=thread_num) ind = thread_num-1;
        level_set[ind].push_back(col);
    }

    return level_set;
}

template<typename T>
std::vector<std::vector<uint32_t>>
TriangularPara<T>::calLevelSetPrune(SparseMatrix<T> &L, std::vector<uint32_t> b_nnz) {
    std::vector<bool> visit(L.m);
    std::vector<uint32_t> reach_set;
    std::vector<std::vector<uint32_t >> level_set;

    for(auto &e:b_nnz){
        dfs(e,L,visit,reach_set);
    }

    std::unordered_map<uint32_t,uint32_t> indeg;
    for(auto &col:reach_set){
        indeg.emplace(col,0);
    }

    for (auto &col:reach_set){
        for (uint32_t j = L.outer_starts[col]+1; j < L.outer_starts[col+1]; j++){
            if(visit[L.inner_indices[j]])
                indeg[L.inner_indices[j]]++;
        }
    }

    level_set.emplace_back(std::vector<uint32_t>());

    for(auto &col:reach_set){
        if(indeg[col] == 0){
            level_set[0].push_back(col);
        }
    }

    uint32_t cur_level = 0;

    while(!level_set[cur_level].empty()){
        cur_level++;
        level_set.emplace_back(std::vector<uint32_t>());
        for(uint32_t &ind:level_set[cur_level-1]){
            for (uint32_t j = L.outer_starts[ind] + 1; j < L.outer_starts[ind+1]; j++){
                uint32_t row = L.inner_indices[j];
                if(!visit[row]) continue;
                indeg[row]--;
                if(indeg[row] == 0){
                    level_set[cur_level].push_back(row);
                }
            }
        }
    }
    return level_set;
}

template<typename T>
void TriangularPara<T>::dfs(uint32_t node,SparseMatrix<T> &L, std::vector<bool> &visit, std::vector<uint32_t> &reach_set) {
    if(visit[node]) return;
    visit[node] = true;
    for(uint32_t ind = L.outer_starts[node]+1;ind<L.outer_starts[node+1];ind++){
        if(!visit[L.inner_indices[ind]]) dfs(L.inner_indices[ind],L,visit,reach_set);
    }
    reach_set.push_back(node);
}

template<typename T>
int TriangularPara<T>::symAnalyze(SparseMatrix<T> &L, std::vector<T> &x, bool verbose) {
    assert(L.m == L.n && L.m == x.size());

    std::vector<uint32_t> nnz;

    if(cal_level == LevelSetType ::Normal){
        level_set = calLevelSet(L);
    }
    else if(cal_level == LevelSetType::Prune){
        std::vector<uint32_t > nnz;
        for(uint32_t row=0;row<x.size();row++){
            if(!isZero(x[row])) nnz.push_back(row);
        }
        level_set = calLevelSetPrune(L,nnz);
    }
    else{
        level_set = calLevelSetAverage(L);
    }

    if(verbose){
//    std::cout<<"Non-zeros in b:"<<nnz.size()<<std::endl;
        std::cout<<"Size of level set:"<<level_set.size()<<std::endl;
        if(cal_level == LevelSetType::Average){
            uint32_t ct = 0;
            for(auto &v:level_set){
                std::cout<<v.size()<<" ";
                ct+=v.size();
            }
            std::cout<<"nnz:"<<L.nnz<<" level ele size: "<<ct<<std::endl;
        }
    }

    return 0;
}

#endif //SPARSE_KERNEL_TRIANGULAR_PARA_H
