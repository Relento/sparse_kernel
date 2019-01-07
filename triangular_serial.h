//
// Created by Rel on 2019/1/6.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_SERIAL_H
#define SPARSE_KERNEL_TRIANGULAR_SERIAL_H

#include "triangular_solver.h"
#include <stack>
#include <utility>
#include "numerical_utils.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;

template <typename T>
class TriangularSerial : public TriangularSolver<T>{
public:
    int solve(SparseMatrix<T> &L, std::vector<T> &x);

    // Calculate the reachable set specified in the algorithm Gilbert(1988)
    // b_nnz is the row ind of non-zero entries in b vector
    // entries are sorted in **reverse** topological order
    std::vector<uint32_t> calReachable(SparseMatrix<T> &L,std::vector<uint32_t> &b_nnz);
    std::vector<uint32_t> calReachable2(SparseMatrix<T> &L,std::vector<uint32_t> &b_nnz);
    void dfs(uint32_t node,SparseMatrix<T> &L,std::vector<bool> &visit,std::vector<uint32_t> &reach_set);
};

template<typename T>
int TriangularSerial<T>::solve(SparseMatrix<T> &L, std::vector<T> &x) {
    if(L.m != L.n || L.m != x.size())
        return -1;

    std::vector<uint32_t> nnz;


    auto t1 = high_resolution_clock::now();
    for(uint32_t i = 0;i<L.m;i++){
        if(!isZero(x[i])) nnz.push_back(i);
    }

    std::cout<<nnz.size()<<std::endl;
    std::vector<uint32_t> reach_set = calReachable(L,nnz);
    std::cout<<reach_set.size()<<std::endl;

    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Calculate reachable Time:" << (double)duration / 1e6 <<std::endl;

    // Calculate entries in reachable set in topological order
    for(auto it = reach_set.rbegin();it != reach_set.rend(); it++){

        uint32_t j = *it;
        // Normalize diagonal entry
        x[j] /= L.values[L.outer_starts[j]];

        for(uint32_t p = L.outer_starts[j]+1 ; p < L.outer_starts[j+1] ; p++){
            x[L.inner_indices[p]] -= L.values[p] * x[j];
        }

    }
    return 0;
}

template<typename T>
std::vector<uint32_t> TriangularSerial<T>::calReachable2(SparseMatrix<T> &L, std::vector<uint32_t> &b_nnz) {

    // A modified non-recursive dfs that can calculate topological order at the same time
    std::vector<bool> visit(L.m);
    std::vector<uint32_t> reach_set;
    std::stack<std::pair<bool,uint32_t>> st;

    for (auto &e:b_nnz){
        st.push(std::make_pair(false,e));
    }
    std::pair<bool,uint32_t> cur;


    while(!st.empty()){
        cur = st.top();
        st.pop();

        if(cur.first){
            // All children have been visited
            reach_set.push_back(cur.second);
            continue;
        }

        visit[cur.second] = true;

        // Mark the end of chilren visiting
        st.push(std::make_pair(true,cur.second));
        for(uint32_t i = L.outer_starts[cur.second];i < L.outer_starts[cur.second+1];i++){
            if(visit[L.inner_indices[i]]) continue;
            st.push(std::make_pair(false,L.inner_indices[i]));
        }

    }

    return reach_set;
}

template<typename T>
std::vector<uint32_t> TriangularSerial<T>::calReachable(SparseMatrix<T> &L, std::vector<uint32_t> &b_nnz) {
    std::vector<bool> visit(L.m);
    std::vector<uint32_t> reach_set;
    for(auto &e:b_nnz){
        dfs(e,L,visit,reach_set);
    }
    return reach_set;
}

template<typename T>
void TriangularSerial<T>::dfs(uint32_t node,SparseMatrix<T> &L, std::vector<bool> &visit, std::vector<uint32_t> &reach_set) {
    if(visit[node]) return;
    visit[node] = true;
    for(uint32_t ind = L.outer_starts[node];ind<L.outer_starts[node+1];ind++){
        if(!visit[L.inner_indices[ind]]) dfs(L.inner_indices[ind],L,visit,reach_set);
    }
    reach_set.push_back(node);
}


#endif //SPARSE_KERNEL_TRIANGULAR_SERIAL_H
