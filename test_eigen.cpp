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

static std::string data_path = "/mnt/c/Users/Rel/Desktop/Toronto/Dehnavi/sparse_kernel/data/";

using namespace std::chrono;
using namespace Eigen;

int main(int argc,char ** argv){
    if(argc >= 2){
        data_path = std::string(argv[1]);
    }
    VectorXd x;
    Eigen::SparseMatrix<double, ColMajor> L_a,L_t,b_mat_sparse,b_torso,L_tso,b_tso;
    VectorXd b_mat_dense;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    assert(loadMarket(b_mat_sparse,data_path+"b_sparse_af_0_k101.mtx"));
    assert(loadMarketVector(b_mat_dense,data_path+"b_dense_af_0_k101.mtx"));
    assert(loadMarket(L_a,data_path+"af_0_k101_lower.mtx"));
    assert(loadMarket(L_t,data_path+"torso1_lower.mtx"));
    assert(loadMarket(b_torso,data_path+"b_for_torso1.mtx"));
    assert(loadMarket(L_tso,data_path+"TSOPF_RS_b678_c2_lower.mtx"));
    assert(loadMarket(b_tso,data_path+"b_for_TSOPF_RS_b678_c2_b.mtx"));

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Load Time:" << (double)duration / 1e6 <<std::endl;


    MatrixXd tmp(b_mat_sparse);
    t1 = high_resolution_clock::now();
    x = L_a.triangularView<Lower>().solve(tmp);
    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;

    std::cout<< "Err norm:"<<(L_a.triangularView<Lower>()*x-b_mat_sparse).norm()<<std::endl;

    MatrixXd tmp2(b_mat_dense);
    t1 = high_resolution_clock::now();
    x = L_a.triangularView<Lower>().solve(tmp2);
    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;
    std::cout<< "Err norm:"<<(L_a.triangularView<Lower>()*x-b_mat_dense).norm()<<std::endl;

    MatrixXd tmp3(b_torso);
    t1 = high_resolution_clock::now();
    x = L_t.triangularView<Lower>().solve(tmp3);
    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;
    std::cout<< "Err norm:"<<(L_t.triangularView<Lower>()*x-b_torso).norm()<<std::endl;

    MatrixXd tmp4(b_tso);
    t1 = high_resolution_clock::now();
    x = L_tso.triangularView<Lower>().solve(tmp4);
    t2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "Solve Time:" << (double)duration / 1e6 <<std::endl;
    std::cout<< "Err norm:"<<(L_tso.triangularView<Lower>()*x-b_tso).norm()<<std::endl;
    return 0;
}

