cmake_minimum_required(VERSION 3.13)
project(sparse_kernel)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

include_directories("C:\\Users\\Rel\\Desktop\\Toronto\\Dehnavi\\eigen_library")

set(COMMON_SOURCES
        utils.h
        sparse_matrix.h
        triangular_tester.h
        triangular_solver.h
        triangular_naive.h
        numerical_utils.h
        triangular_serial.h;
        )

#add_executable(main main.cpp ${COMMON_SOURCES})

add_executable(test_main test_main.cpp ${COMMON_SOURCES} )
add_executable(test_eigen test_eigen.cpp ${COMMON_SOURCES} )
