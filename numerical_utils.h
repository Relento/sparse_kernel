//
// Created by Rel on 2019/1/7.
//

#ifndef SPARSE_KERNEL_NUMERICAL_UTILS_H
#define SPARSE_KERNEL_NUMERICAL_UTILS_H

#include <vector>
#include <cmath>

const double eps = 1e-45;
template <typename T>
bool inline isZero(T value);

template <>
bool inline isZero(double value){
    return std::fabs(value)<eps;
}

// Calculate 2-norm(vec1-vec2)
template <typename T>
T differenceNorm(const std::vector<T> &vec1,const std::vector<T> &vec2);

template <>
double differenceNorm(const std::vector<double> &vec1,const std::vector<double> &vec2){
    assert(vec1.size() == vec2.size());

    double res = 0;
    for(uint32_t i = 0; i < vec1.size();i++){
        res+=(vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
    }
    return sqrt(res);
};

#endif //SPARSE_KERNEL_NUMERICAL_UTILS_H
