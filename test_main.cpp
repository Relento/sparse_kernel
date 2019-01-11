//
// Created by Rel on 2019/1/5.
//

#include "triangular_tester.h"
#include "triangular_naive.h"
#include "triangular_serial.h"
#include "triangular_para.h"
#include "utils.h"
#include <chrono>
#include <fstream>

static std::string data_path = "C:\\Users\\Rel\\Desktop\\Toronto\\Dehnavi\\sparse_kernel\\data\\";

using namespace std::chrono;

int main(int argc,char **argv) {
    bool test_big = false;
    if(argc>=2 && std::string(argv[1]) == "--testbig") test_big = true;
    if(argc>=3) data_path = std::string(argv[2]);

    SparseMatrix<double> L, b_mat,L2,L3,b_mat_sparse,b_mat_torso1,b_mat_tso,L4;

    std::vector<double> b,b_sparse,b_dense,b_torso1,b_tso;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    assert(loadMatrix(L, data_path + "10.mtx",false)==0);
    assert(loadMatrix(b_mat, data_path + "b_10.mtx",false)==0);

    if(test_big){
        assert(loadMatrix(b_mat_sparse, data_path + "b_sparse_af_0_k101.mtx",true)==0);
        assert(loadVector(b_dense, data_path + "b_dense_af_0_k101.mtx")==0);
//        assert(loadMatrix(L2, data_path + "af_0_k101_lower.mtx",true)==0);
        assert(loadMatrix(L2, data_path + "af_0_k101.mtx",true)==0);

        // Torso1
//        assert(loadMatrix(L3, data_path + "torso1_lower.mtx",true)==0);
        assert(loadMatrix(L3, data_path + "torso1.mtx",true)==0);
        assert(loadMatrix(b_mat_torso1, data_path + "b_for_torso1.mtx",true)==0);

        // TSOPF
//        assert(loadMatrix(L4, data_path + "TSOPF_RS_b678_c2_lower.mtx",true)==0);
        assert(loadMatrix(L4, data_path + "TSOPF_RS_b678_c2.mtx",true)==0);
        assert(loadMatrix(b_mat_tso, data_path + "b_for_TSOPF_RS_b678_c2_b.mtx",true)==0);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Load Time:" << (double) duration / 1e6 << "s" << std::endl;

    b_mat.col2Vec(0, b);
    b_mat_sparse.col2Vec(0,b_sparse);
    b_mat_torso1.col2Vec(0,b_torso1);
    b_mat_tso.col2Vec(0,b_tso);


    TriangularNaive<double> naive;
    TriangularSerial<double> serial;
    TriangularPara<double> para;

    TriangularTester<double> tester;


    if(test_big){
        tester.loadCase(&L2, &b_sparse,"sparse");
        tester.loadCase(&L2, &b_dense,"dense");
        tester.loadCase(&L3, &b_torso1,"torso1");
        tester.loadCase(&L4, &b_tso,"tsopf");
    }
    else{
        tester.loadCase(&L, &b,"10");
    }
    std::ofstream f_save("res.txt");
    std::vector<TriangularTester<double>::TestResult> test_results;

    test_results = tester.testSolver(&naive,"naive",true,data_path);
    tester.prettyPrintResult(f_save,test_results);

    test_results = tester.testSolver(&serial,"serial",true,data_path);
    tester.prettyPrintResult(f_save,test_results);

    para.cal_level = TriangularPara<double>::LevelSetType::Normal;
    test_results = tester.testSolver(&para,"para_normal",true,"");
    tester.prettyPrintResult(f_save,test_results);

    para.cal_level = TriangularPara<double>::LevelSetType::Prune;
    test_results = tester.testSolver(&para,"para_prune",true,"");
    tester.prettyPrintResult(f_save,test_results);

    para.cal_level = TriangularPara<double>::LevelSetType::Average;
    test_results = tester.testSolver(&para,"para_average",true,"");
    tester.prettyPrintResult(f_save,test_results);

    return 0;

}
