//
// Created by Rel on 2019/1/5.
//

#include "triangular_tester.h"
#include "triangular_naive.h"
#include "triangular_serial.h"
#include "triangular_para.h"
#include "utils.h"
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

static std::string data_path = "C:\\Users\\Rel\\Desktop\\Toronto\\Dehnavi\\sparse_kernel\\data\\";

using namespace std::chrono;

int main(int argc,char **argv) {
    bool test_big = false;
    if(argc>=2 && std::string(argv[1]) == "--testbig") test_big = true;
    if(argc>=3) data_path = std::string(argv[2]);

    SparseMatrix<double> L, b_mat,L2,L3,b_mat_sparse,b_mat_torso1;

    std::vector<double> b,b_sparse,b_dense,b_torso1;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    assert(loadMatrix(L, data_path + "10.mtx")==0);
    assert(loadMatrix(b_mat, data_path + "b_10.mtx")==0);

    if(test_big){
        assert(loadMatrix(b_mat_sparse, data_path + "b_sparse_af_0_k101.mtx")==0);
        assert(loadVector(b_dense, data_path + "b_dense_af_0_k101.mtx")==0);
        assert(loadMatrix(L2, data_path + "af_0_k101_lower.mtx")==0);

        // Torso1
        assert(loadMatrix(L3, data_path + "torso1_lower.mtx")==0);
        assert(loadMatrix(b_mat_torso1, data_path + "b_for_torso1.mtx")==0);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Load Time:" << (double) duration / 1e6 << "s" << std::endl;

    b_mat.col2Vec(0, b);
    b_mat_sparse.col2Vec(0,b_sparse);
    b_mat_torso1.col2Vec(0,b_torso1);


    TriangularNaive<double> naive;
    TriangularSerial<double> serial;
    TriangularPara<double> para;

    TriangularTester<double> tester;


    if(test_big){
        tester.loadCase(&L2, &b_sparse,"sparse");
        tester.loadCase(&L2, &b_dense,"dense");
        tester.loadCase(&L3, &b_torso1,"torso1");
    }
    else{
        tester.loadCase(&L, &b,"10");
    }

//    tester.testSolver(&naive,"naive",true,data_path);
//    tester.testSolver(&serial,"serial",true,data_path);

//    tester.testSolver(&naive,"naive",true,"");
//    tester.testSolver(&serial,"serial",true,"");
    tester.testSolver(&para,"para",true,"");

    return 0;

}
