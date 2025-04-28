#include <gtest/gtest.h>                  // Google Test framework
#include "R_spanner_kernels.cuh"          

TEST(ComputeRowPtrTest, SimplePrefixSum)
{
    std::vector<long long> input = {0, 1, 2, 0};   
    std::vector<long long> output(4);             

    compute_row_ptr(input, output, 4);      

    std::vector<long long> expected = {0, 0, 1, 3}; 
    ASSERT_EQ(output, expected);             
}
