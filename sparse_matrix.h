//
// Created by Rel on 2019/1/5.
//

#ifndef SPARSE_KERNEL_SPARSE_MATRIX_H
#define SPARSE_KERNEL_SPARSE_MATRIX_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cassert>
#include "numerical_utils.h"


template<typename T>
class SparseMatrix {
public:
    SparseMatrix() = default ;
    // Initialize from a column vector
    SparseMatrix(const std::vector<T> &vec);
    SparseMatrix(uint32_t m,uint32_t n,uint32_t nnz,
            const std::vector<T> &values,const std::vector<uint32_t> &inner_indices,const std::vector<uint32_t > &outer_starts)
            :m(m),n(n),nnz(nnz),values(values),inner_indices(inner_indices),outer_starts(outer_starts){}


    void display();
    // Convert a column to vector
    int col2Vec(uint32_t col,std::vector<T> &vec);
    std::vector<T> operator *(const std::vector<T> &x);
//private:

    std::vector<T> values;
    std::vector<uint32_t> inner_indices,outer_starts;

    // m: row size, n: column size, nnz: number of non-zero entires
    uint32_t m = 0,n = 0,nnz = 0;

};

template <typename T>
void SparseMatrix<T>::display() {
    uint32_t  nnz = 0;
    // display the transpose of the matrix
    for (uint32_t col = 0; col < n;col++){
        for (uint32_t row = 0; row < m;row++){
            if (nnz == outer_starts[col+1] || row < inner_indices[nnz]){
                std::cout<<"0 ";
                continue;
            }
            std::cout<<values[nnz]<<" ";
            nnz++;
        }
        std::cout<<std::endl;
    }

    std::cout<<std::endl;
    std::cout<<std::endl;

    for (auto x:inner_indices){
        std::cout<<x<<" ";
    }
    std::cout<<std::endl;

    for (auto x:outer_starts){
        std::cout<<x<<" ";
    }

    std::cout<<std::endl;

    for (auto x:values){
        std::cout<<x<<" ";
    }

    std::cout<<std::endl;
}

template<typename T>
int SparseMatrix<T>::col2Vec(uint32_t col, std::vector<T> &vec) {
    if(col >= n){
        return -1;
    }
    vec = std::vector<T>(m);
    for (uint32_t i = outer_starts[col];i<outer_starts[col+1];i++){
        vec[inner_indices[i]] = values[i];
    }
    return 0;
}

template<typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<T> &vec) {
    m = vec.size();
    n = 1;
    nnz = 0;
    inner_indices = std::vector<uint32_t>();
    values = std::vector<T>();
    for (uint32_t row = 0;row<m;row++){
        T x = vec[row];
        if(!isZero(x)){
            nnz++;
            inner_indices.push_back(row);
            values.push_back(x);
        }
    }
    outer_starts = std::vector<uint32_t>({0,nnz});
}

template<typename T>
std::vector<T> SparseMatrix<T>::operator*(const std::vector<T> &x) {
    assert(x.size() == n);

    std::vector<T> res(m); // Should be initialized to zero(true for all known numerical types)
    for (uint32_t col = 0;col<n;col++){
        for (uint32_t ind = outer_starts[col];ind < outer_starts[col+1];ind++){
            res[inner_indices[ind]]+=values[ind]*x[col];
        }
    }

    return res;
}




#endif //SPARSE_KERNEL_SPARSE_MATRIX_H
