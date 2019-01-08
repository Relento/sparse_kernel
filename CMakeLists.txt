cmake_minimum_required(VERSION 3.13)
project(sparse_kernel)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")

if (WIN32)
include_directories("C:\\Users\\Rel\\Desktop\\Toronto\\Dehnavi\\eigen_library")
endif (WIN32)

if (UNIX)
include_directories("/home/rel/Desktop/eigen-git-mirror")
endif()


set(COMMON_SOURCES
        utils.h
        sparse_matrix.h
        triangular_tester.h
        triangular_solver.h
        triangular_naive.h
        numerical_utils.h
        triangular_serial.h;
        triangular_para.h)

#add_executable(main main.cpp ${COMMON_SOURCES})

add_executable(test_main test_main.cpp ${COMMON_SOURCES} )
add_executable(test_eigen test_eigen.cpp ${COMMON_SOURCES} )
