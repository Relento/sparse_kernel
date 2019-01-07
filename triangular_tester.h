//
// Created by Rel on 2019/1/5.
//

#ifndef SPARSE_KERNEL_TRIANGULAR_TEST_H
#define SPARSE_KERNEL_TRIANGULAR_TEST_H


#include "triangular_solver.h"
#include "numerical_utils.h"
#include "utils.h"
#include <chrono>
#include <cmath>

using namespace std::chrono;

template <typename T>
class TriangularTester{
public:
    class TestResult;

    // list of testcases
    std::vector<SparseMatrix<T> *> Ls;
    std::vector<std::vector<T> *> bs;
    std::vector<std::string> names;

    void loadCase(SparseMatrix<T> * L,std::vector<T>* b,std::string name){
        Ls.push_back(L);
        bs.push_back(b);
        names.push_back(name);
    }

    // if save_path is not empty, save each solution to corresponding file
    // NOTE: save_path should end with backslash
    std::vector<TestResult> testSolver(TriangularSolver<T> *s,
            const std::string &solver_name, bool cal_err = true, const std::string &save_path = "");

    // Store the result of a testcase
    struct TestResult{
        T err_norm;
        double duration;
        TestResult(T err_norm, double duration):err_norm(err_norm),duration(duration){}
    };
};

template<>
std::vector<TriangularTester<double>::TestResult>
TriangularTester<double>::testSolver(TriangularSolver<double> *solver,
        const std::string &solver_name,bool cal_err, const std::string &save_path) {

    std::vector<TestResult> res;
    high_resolution_clock::time_point t1,t2;
    SparseMatrix<double> L_dup;
    double err_norm;

    std::cout<<"Testing "<<solver_name<<std::endl;
    for (uint32_t i = 0; i < Ls.size();i++){
        auto &L = *(Ls[i]);
        auto &b = *(bs[i]);
        std::vector<double> x(b);

        std::cout<<std::endl;

        //Duplicate L in case the solver modifies the original
        // which influences the following test
        L_dup = L;

        std::cout.precision(6);
        std::cout<<"Test case \""<<names[i]<<"\":"<<std::endl;
        std::cout<<"\tCol size: "<<Ls[i]->n<<std::endl;

        t1 = high_resolution_clock::now();
        int err = solver->solve(L_dup,x);
        t2 = high_resolution_clock::now();

        if (err!=0){
            std::cerr<<"Something wrong happens! "<<std::endl;
            continue;
        }

        auto duration = duration_cast<microseconds>(t2-t1).count();

        if (cal_err){
            err_norm = differenceNorm(L_dup*x,b);
        }

        else{
            err_norm = -1;
        }

        std::cout<<"\tSolve time:"<<((double)duration)/1e6<<"s"<<std::endl;
        std::cout<<"\tError norm:"<<err_norm<<std::endl;
        res.emplace_back(err_norm,(double)duration/1e6);


        uint32_t nnz45 = 0;
        uint32_t nnz50 = 0;
        uint32_t nnz55 = 0;
        for (auto &e: x){
            if(fabs(e)>1e-50) nnz50++;
            if(fabs(e)>1e-45) nnz45++;
            if(fabs(e)>1e-55) nnz55++;
        }
        std::cout<<nnz45<<" "<<nnz50<<" "<<nnz55<<std::endl;
        if(!save_path.empty()){
            saveMatrix(SparseMatrix<double>(x),save_path+names[i]+"_"+solver_name+".mtx");
        }
    }

    return res;
}


#endif //SPARSE_KERNEL_TRIANGULAR_TEST_H
