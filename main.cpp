#include <iostream>
#include <string>
#include <chrono>

#include "sparse_matrix.h"
#include "utils.h"


static std::string data_path = "C:\\Users\\Rel\\Desktop\\Toronto\\Dehnavi\\sparse_kernel\\data\\";

using namespace std::chrono;

int main() {
    SparseMatrix<double> mat;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    loadMatrix(mat,data_path+"small_low.mtx");
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    std::cout << "Time:" << duration<<std::endl;

    std::cout<<std::endl;
    mat.display();
    std::cout<<std::endl;
    saveMatrix(mat,data_path+"save.mtx");
    return 0;
}