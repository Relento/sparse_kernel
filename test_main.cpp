//
// Created by Rel on 2019/1/5.
//

#include "triangular_tester.h"
#include "triangular_naive.h"
#include "utils.h"
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

static std::string data_path = "C:\\Users\\Rel\\Desktop\\Toronto\\Dehnavi\\sparse_kernel\\data\\";

using namespace std::chrono;

int main() {
    SparseMatrix<double> L, b_mat,L2,b_mat_sparse;

    std::vector<double> b,b_sparse,b_dense;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    assert(loadMatrix(L, data_path + "10.mtx")==0);
    assert(loadMatrix(b_mat, data_path + "b_10.mtx")==0);

//#define TESTBIG
#ifdef TESTBIG
    assert(loadMatrix(b_mat_sparse, data_path + "b_sparse_af_0_k101.mtx")==0);
    assert(loadVector(b_dense, data_path + "b_dense_af_0_k101.mtx")==0);
    assert(loadMatrix(L2, data_path + "af_0_k101_lower.mtx")==0);
#endif
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Load Time:" << (double) duration / 1e6 << "s" << std::endl;

    b_mat.col2Vec(0, b);
    b_mat_sparse.col2Vec(0,b_sparse);

    TriangularNaive<double> naive;
    TriangularTester<double> tester;
    tester.loadCase(&L, &b,"10");

#ifdef TESTBIG
    tester.loadCase(&L2, &b_sparse,"sparse");
    tester.loadCase(&L2, &b_dense,"dense");
#endif

    tester.testSolver(&naive,"naive",true,data_path);

    return 0;

}
