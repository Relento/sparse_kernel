//
// Created by Rel on 2019/1/5.
//

#ifndef SPARSE_KERNEL_UTILS_H
#define SPARSE_KERNEL_UTILS_H

#include "sparse_matrix.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <tuple>
#include <algorithm>

enum class MatrixFormat {MATRIXMARKET};

// for Matrix Market format, currently only support coordinate real generate type
template <typename T>
int loadMatrix(SparseMatrix<T> &mat,const std::string &file_name,bool only_lower = false,MatrixFormat mat_format = MatrixFormat::MATRIXMARKET){

    std::ifstream fin(file_name,std::ios::in);
    if (!fin){
        std::cerr<<"Cannot open the file:"<<file_name<<std::endl;
        return -1;
    }
    uint32_t m,n,nnz;
    std::string line;

    int line_ct  = 1;
    std::getline(fin,line);
    while (!line.empty() && line[0] == '%'){
        std::getline(fin,line);
        line_ct++;
    }

    std::istringstream line_stream(line);
    line_stream>>m>>n>>nnz;

    typedef std::tuple<uint32_t ,uint32_t ,T> Entry;
    std::vector<Entry> entries(nnz);

    for(uint32_t i=0;i<nnz;i++){
        std::getline(fin,line);
        line_ct++;
        if(line.empty()){
            std::cerr<<"Unexpected EOF"<<std::endl;
            return -1;
        }

        line_stream = std::istringstream(line);
        uint32_t row,col;
        T value;

        if(!(line_stream>>row>>col>>value)){
            std::cerr<<file_name<<":line:"<<line_ct<<": Error while reading a entry!"<<std::endl;
            return -1;
        }

        //
        entries[i] = std::make_tuple(row-1,col-1,value); // For MM format, index starts from 1
    }
//#define LOAD_TUPLE_SORT

// Sort the entries according to column for CSC storation
#ifdef LOAD_TUPLE_SORT
    std::sort(entries.begin(),entries.end(),
              [](const Entry &x,const Entry &y)->bool{
                  if (std::get<1>(x) == std::get<1>(y))
                    return std::get<0>(x) < std::get<0> (y);
                  else return std::get<1>(x) < std::get<1>(y);
              });
#endif

    std::vector<T> values(nnz);
    std::vector<uint32_t> inner_indices(nnz),outer_starts(n);
    outer_starts[0] = 0;

    uint32_t cur_col = 0;
    for(uint32_t i=0;i<nnz;i++){
        Entry &entry = entries[i];
        values[i] = std::get<2>(entry);
        uint32_t x = std::get<0>(entry),y = std::get<1>(entry);
        inner_indices[i] = x;

        while (y > cur_col){
            cur_col++;
            outer_starts[cur_col] = i;
        }
    }
    outer_starts.push_back(nnz);
    mat = SparseMatrix<T>(m,n,nnz,values,inner_indices,outer_starts);
    return 0;
}

template <typename T>
int saveMatrix(const SparseMatrix<T> &mat,const std::string &file_name,MatrixFormat mat_format = MatrixFormat::MATRIXMARKET){
    std::ofstream fout(file_name);
    if (!fout){
        std::cerr<<"Unable to open file:"<<file_name<<std::endl;
        return -1;
    }
    fout.flags(std::ios_base::scientific);
    fout.precision(64);
    fout<<"%%MatrixMarket matrix coordinate real general"<<std::endl;
    fout<<mat.m<<" "<<mat.n<<" "<<mat.nnz<<std::endl;
    for (uint32_t col = 0;col<mat.n;col++){
        for(uint32_t row_ind = mat.outer_starts[col];row_ind < mat.outer_starts[col+1];row_ind++){
            fout<<mat.inner_indices[row_ind]+1<<" "<<col+1<<" "<<mat.values[row_ind]<<std::endl;
        }
    }

    return 0;
}

// read a column vector from  Matrix Market format, only support array real generate type
template <typename T>
int loadVector(std::vector<T> &vec,const std::string file_name){
    std::ifstream fin(file_name);
    if (!fin){
        std::cerr<<"Cannot open the file:"<<file_name<<std::endl;
        return -1;
    }
    uint32_t m,n;
    std::string line;

    int line_ct  = 1;
    std::getline(fin,line);
    while (!line.empty() && line[0] == '%'){
        std::getline(fin,line);
        line_ct++;
    }

    std::istringstream line_stream(line);
    line_stream>>m>>n;

    if (n!=1){
        std::cerr<<"The file doesn't store a column vector!"<<file_name<<std::endl;
        return -1;
    }

    vec = std::vector<T>(m);

    for(uint32_t i=0;i<m;i++) {
        std::getline(fin, line);
        line_ct++;
        if (line.empty()) {
            std::cerr << "Unexpected EOF" << std::endl;
            return -1;
        }

        line_stream = std::istringstream(line);

        T value;

        if (!(line_stream >> value)) {
            std::cerr << file_name << ":line:" << line_ct << ": Error while reading a entry!" << std::endl;
            return -1;
        }

        vec[i] = value;
    }

    return 0;
}


// check if the input is a band-diagonal matrix
template <typename T>
bool isBand(SparseMatrix<T> L){
    for(uint col = 0;col <L.n;col++){
        uint32_t prev;
        for(uint32_t j = L.outer_starts[col] ; j < L.outer_starts[col+1] ; j++){
            if(j == L.outer_starts[col]) {
                prev = L.inner_indices[L.outer_starts[col]];
                continue;
            }
            uint32_t cur = L.inner_indices[j];
            if(j!= prev + 1){
                return false;
            }
            prev = cur;
        }
    }
    return true;
}
#endif //SPARSE_KERNEL_UTILS_H
