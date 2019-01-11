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
#include <cstdint>

using namespace std::chrono;

template <typename T>
class TriangularTester{
public:
    class TestResult;

    // list of testcases
    std::vector<SparseMatrix<T> *> Ls;
    std::vector<std::vector<T> *> bs;
    std::vector<std::string> names;
    uint32_t test_ct = 10;

    void loadCase(SparseMatrix<T> * L,std::vector<T>* b,std::string name){
        Ls.push_back(L);
        bs.push_back(b);
        names.push_back(name);
    }

    // if save_path is not empty, save each solution to corresponding file
    // NOTE: save_path should end with backslash
    std::vector<TestResult> testSolver(TriangularSolver<T> *s,
            const std::string &solver_name, bool cal_err = true, const std::string &save_path = "", bool verbose = false);

    // Store the result of a testcase
    struct TestResult{
        std::string solver_name;
        std::string testcase_name;
        T err_norm;
        double sym_time,solve_time;
        TestResult(const std::string &solver_name,const std::string &testcase_name,T err_norm, double sym_time,double solve_time)
        :solver_name(solver_name),testcase_name(testcase_name),err_norm(err_norm),sym_time(sym_time),solve_time(solve_time){}
    };

    void prettyPrintResult(std::ostream &is,const std::vector<TestResult> ts);
};

template<>
std::vector<TriangularTester<double>::TestResult>
TriangularTester<double>::testSolver(TriangularSolver<double> *solver,
        const std::string &solver_name,bool cal_err, const std::string &save_path,bool verbose ) {

    std::vector<TestResult> res;
    high_resolution_clock::time_point t1,t2;
    SparseMatrix<double> L_dup;
    double err_norm,time_solve = 0,time_sym = 0;

    if(verbose){
        std::cout<<"Testing "<<solver_name<<std::endl;
    }

    for (uint32_t i = 0; i < Ls.size();i++){
        auto &L = *(Ls[i]);
        auto &b = *(bs[i]);
        std::vector<double> x;


        //Duplicate L in case the solver modifies the original
        // which influences the following test
        L_dup = L;

        std::cout.precision(6);
        if(verbose){
            std::cout<<"Test case \""<<names[i]<<"\":"<<std::endl;
            std::cout<<"\tCol size: "<<Ls[i]->n<<std::endl;
        }

        int err;
        for(uint32_t i=0;i<test_ct;i++){
            x = b;

            // Test symbolic analysis
            t1 = high_resolution_clock::now();
            err = solver->symAnalyze(L_dup,x,verbose);
            t2 = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(t2-t1).count();
            time_sym+=(double)duration/10e6;
            if(err!=0) break;

            // Test numerical solve
            t1 = high_resolution_clock::now();
            err = solver->solve(L_dup,x,verbose);
            t2 = high_resolution_clock::now();
            duration = duration_cast<microseconds>(t2-t1).count();
            time_solve+=(double)duration/10e6;
            if(err!=0) break;
        }

        time_sym/=test_ct;
        time_solve/=test_ct;

        if (err!=0){
            std::cerr<<"Something wrong happens! "<<std::endl;
            continue;
        }


        // Verify the result by calculating the norm of the error term
        if (cal_err){
            err_norm = differenceNorm(L_dup*x,b);
        }

        else{
            err_norm = -1;
        }

        if(verbose){
            std::cout<<"\tSymbolic Analysis time:"<<time_sym<<"s"<<std::endl;
            std::cout<<"\tNumerical Computation time:"<<time_solve<<"s"<<std::endl;
            if(cal_err)
                std::cout<<"\tError norm:"<<err_norm<<std::endl;
        }
        res.emplace_back(solver_name,names[i],err_norm,time_sym,time_solve);

        if(!save_path.empty()){
            saveMatrix(SparseMatrix<double>(x),save_path+names[i]+"_"+solver_name+".mtx");
        }

    }
    if(verbose){
        std::cout<<std::endl;
    }

    return res;
}

template<typename T>
void TriangularTester<T>::prettyPrintResult(std::ostream &is, const std::vector<TriangularTester::TestResult> ts) {
    assert(!ts.empty());
    is<<ts[0].solver_name<<"\tErr\tSymbolic\tNumerical\tTotal"<<std::endl;
    for(auto &t:ts){
        is<<t.testcase_name<<"\t"<<t.err_norm<<"\t"<<t.sym_time<<"\t"<<t.solve_time<<"\t";
        is<<t.sym_time+t.solve_time<<std::endl;
    }
    is<<std::endl;
    return;
}


#endif //SPARSE_KERNEL_TRIANGULAR_TEST_H
