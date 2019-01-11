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
    enum class CalReachMethod  {REC_DFS,NON_REC_DFS};

    CalReachMethod cal_reach = CalReachMethod ::NON_REC_DFS;

    std::vector<uint32_t> reach_set;

    int solve(SparseMatrix<T> &L, std::vector<T> &x,bool verbose = false);
    int symAnalyze(SparseMatrix<T> &L, std::vector<T> &x,bool verbose = false);

    // Calculate the reachable set specified in the algorithm Gilbert(1988)
    // b_nnz is the row ind of non-zero entries in b vector
    // entries are sorted in **reverse** topological order

    // Recursive DFS version
    std::vector<uint32_t> calReachableRec(SparseMatrix<T> &L,std::vector<uint32_t> &b_nnz);
    // Non-recursive DFS version
    std::vector<uint32_t> calReachableNonRec(SparseMatrix<T> &L,std::vector<uint32_t> &b_nnz);

    void dfs(uint32_t node,SparseMatrix<T> &L,std::vector<bool> &visit,std::vector<uint32_t> &reach_set);
};

template<typename T>
int TriangularSerial<T>::solve(SparseMatrix<T> &L, std::vector<T> &x, bool verbose) {

    // Calculate entries in reachable set in topological order
    for(auto it = reach_set.rbegin();it != reach_set.rend(); it++){

        uint32_t j = *it;
        // Normalize diagonal entry
        x[j] /= L.values[L.outer_starts[j]];

//#pragma omp parallel for
        for(uint32_t p = L.outer_starts[j]+1 ; p < L.outer_starts[j+1] ; p++){
            x[L.inner_indices[p]] -= L.values[p] * x[j];
        }

    }

    return 0;
}

template<typename T>
std::vector<uint32_t> TriangularSerial<T>::calReachableNonRec(SparseMatrix<T> &L, std::vector<uint32_t> &b_nnz){

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

        if(visit[cur.second]) continue;

        visit[cur.second] = true;

        // Mark the end of chilren visiting
        st.emplace(std::make_pair(true,cur.second));
        for(uint32_t i = L.outer_starts[cur.second] + 1;i < L.outer_starts[cur.second+1];i++){
            if(visit[L.inner_indices[i]]) continue;
            st.push(std::make_pair(false,L.inner_indices[i]));
        }

    }

    return reach_set;
}

template<typename T>
std::vector<uint32_t> TriangularSerial<T>::calReachableRec(SparseMatrix<T> &L, std::vector<uint32_t> &b_nnz){
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
    for(uint32_t ind = L.outer_starts[node]+1;ind<L.outer_starts[node+1];ind++){
        if(!visit[L.inner_indices[ind]]) dfs(L.inner_indices[ind],L,visit,reach_set);
    }
    reach_set.push_back(node);
}

template<typename T>
int TriangularSerial<T>::symAnalyze(SparseMatrix<T> &L, std::vector<T> &x, bool verbose) {
    assert(L.m == L.n && L.m == x.size());

    std::vector<uint32_t> nnz;

    for(uint32_t i = 0;i<L.m;i++){
        if(!isZero(x[i])) nnz.push_back(i);
    }

    if(verbose){
        std::cout<<"Non-zeros in b:"<<nnz.size()<<std::endl;
    }

    if(cal_reach == CalReachMethod::NON_REC_DFS){
        reach_set = calReachableNonRec(L,nnz);
    }
    else{
        reach_set = calReachableRec(L,nnz);
    }

    if(verbose){
        std::cout<<"Size of reachable set:"<<reach_set.size()<<std::endl;
    }
    return 0;
}


#endif //SPARSE_KERNEL_TRIANGULAR_SERIAL_H
