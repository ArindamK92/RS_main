cmake_minimum_required(VERSION 3.18)
project(RSpannerCUDA LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Detect GPU arch automatically
# set(CMAKE_CUDA_ARCHITECTURES "ALL")
# set(CMAKE_CUDA_ARCHITECTURES 75)  # For Turing (RTX 20xx)
set(CMAKE_CUDA_ARCHITECTURES 80)  # For Ampere (A100)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SRC_FILES src/*.cpp src/*.cu)

add_executable(r_spanner_cuda ${SRC_FILES})

# Required for Thrust to compile properly
target_compile_options(r_spanner_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_compile_definitions(r_spanner_cuda PRIVATE $<$<CONFIG:Debug>:DEBUG>)


# Unit Tests
option(ENABLE_TESTS "Enable building unit tests" ON)

if(ENABLE_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
    )
    FetchContent_MakeAvailable(googletest)

    # file(GLOB TEST_FILES tests/*.cpp src/*.cpp src/*.cu)

    add_executable(unit_tests
    tests/test_compute_row_ptr.cpp
    tests/test_filter_kernel.cpp
    #tests/test_find_ngbr_comm_kernel.cpp
    src/R_spanner_kernels.cu
    src/R_spanner_helper.cpp
    src/printer.cpp
    )
    target_link_libraries(unit_tests gtest_main)
    target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)

    add_test(NAME AllUnitTests COMMAND unit_tests)
endif()

