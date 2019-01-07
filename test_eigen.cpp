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
using namespace Eigen;

int main(){
    VectorXd x;
    Eigen::SparseMatrix<double, ColMajor> L,bMat;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
//    assert(loadMarket(L,data_path+"torso2_lower.mtx"));
//    assert(loadMarket(L,data_path+"torso1.mtx"));
//    assert(loadMarket(bMat,data_path+"b_for_torso1.mtx"));
    assert(loadMarket(bMat,data_path+"b_for_TSOPF_RS_b678_c2_b.mtx"));
    assert(loadMarket(L,data_path+"TSOPF_RS_b678_c2.mtx"));
//    assert(loadMarket(L,data_path+"10.mtx"));
//    assert(loadMarket(bMat,data_path+"b_10.mtx"));

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Load Time:" << (double)duration / 1e6 <<std::endl;

    SparseLU<Eigen::SparseMatrix<double, ColMajor>, COLAMDOrdering<uint32_t>> solver;
    t1 = high_resolution_clock::now();

    /*
    // fill A and b;
    // Compute the ordering permutation vector from the structural pattern of A
    solver.analyzePattern(L);
    // Compute the numerical factorization
    solver.factorize(L);
    //Use the factors to solve the linear system
    x = solver.solve(b);
    t2 = high_resolution_clock::now();

     */
    MatrixXd tmp(bMat);
//    std::cout<<L.size()<<" "<<tmp.size()<<std::endl;
    x = L.triangularView<Lower>().solve(MatrixXd(bMat));

    t2 = high_resolution_clock::now();



    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;

//    saveMarket(SparseVector<double,ColMajor>(x.sparseView()),data_path+"eigen_torso1.mtx");
//    saveMarket(SparseVector<double,ColMajor>(x.sparseView()),data_path+"eigen_10.mtx");
    saveMarket(SparseVector<double,ColMajor>(x.sparseView()),data_path+"eigen_tsopf.mtx");

    std::cout<<(L.triangularView<Lower>()*x-bMat).norm()<<std::endl;

    return 0;
}

#if 0
int main(){
    TriangularNaive<double> naive;
    SparseMatrix<double> mat,mat2;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
//    loadMatrix(mat,data_path+"10.mtx");
//    loadMatrix(mat2,data_path+"b_10.mtx");
    loadMatrix(mat,data_path+"TSOPF_RS_b678_c2_lower.mtx");
    loadMatrix(mat2,data_path+"b_for_TSOPF_RS_b678_c2_b.mtx");
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Load Time:" << (double)duration / 1e6 <<std::endl;

    std::vector<double> b;
    mat2.col2Vec(0,b);
    std::vector<double> x(b);


    t1 = high_resolution_clock::now();
    naive.solve(mat,x);
    t2 = high_resolution_clock::now();

    duration = duration_cast<microseconds>( t2 - t1 ).count();

    std::cout << "Solve Time:" << (double)duration/1e6<<std::endl;

    SparseMatrix<double> mat3(x);
//    saveMatrix(mat3,data_path+"naive_10.mtx");
    saveMatrix(mat3,data_path+"naive_tsopf.mtx");

}
#endif
