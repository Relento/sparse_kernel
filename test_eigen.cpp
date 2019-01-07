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
    Eigen::SparseMatrix<double, ColMajor> L,b_mat_sparse;
    VectorXd b_mat_dense;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
//    assert(loadMarket(L,data_path+"torso2_lower.mtx"));
//    assert(loadMarket(L,data_path+"torso1.mtx"));
//    assert(loadMarket(bMat,data_path+"b_for_torso1.mtx"));
//    assert(loadMarket(bMat,data_path+"b_for_TSOPF_RS_b678_c2_b.mtx"));
//    assert(loadMarket(L,data_path+"TSOPF_RS_b678_c2.mtx"));
//    assert(loadMarket(L,data_path+"10.mtx"));
//    assert(loadMarket(bMat,data_path+"b_10.mtx"));
    assert(loadMarket(b_mat_sparse,data_path+"b_sparse_af_0_k101.mtx"));
    assert(loadMarketVector(b_mat_dense,data_path+"b_dense_af_0_k101.mtx"));
    assert(loadMarket(L,data_path+"af_0_k101.mtx"));

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Load Time:" << (double)duration / 1e6 <<std::endl;


    t1 = high_resolution_clock::now();
    x = L.triangularView<Lower>().solve(MatrixXd(b_mat_sparse));
    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;
    std::cout<<(L.triangularView<Lower>()*x-b_mat_sparse).norm()<<std::endl;
//    saveMarket(SparseVector<double,ColMajor>(x.sparseView()),data_path+"eigen_tsopf.mtx");


    t1 = high_resolution_clock::now();
    x = L.triangularView<Lower>().solve(MatrixXd(b_mat_dense));
    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;
    std::cout<< "Err norm:"<<(L.triangularView<Lower>()*x-b_mat_dense).norm()<<std::endl;
//    saveMarket(SparseVector<double,ColMajor>(x.sparseView()),data_path+"eigen_tsopf.mtx");

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
