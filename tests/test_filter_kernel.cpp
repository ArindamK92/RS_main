#include <gtest/gtest.h>
#include "R_spanner_kernels.cuh"

TEST(FilterKernelTest, DetectsBorderVertex)
{
    std::vector<long long> row_ptr = {0, 2, 5, 7, 9};
    std::vector<long long> col_idx = {
        1, 2,        // neighbors of node 0 -> comms 2, 3
        0, 2, 3,     // neighbors of node 1 -> comms 1, 3, 4
        0, 1,        // neighbors of node 2
        1, 2         // neighbors of node 3
    };
    std::vector<long long> C = {1, 2, 3, 4};

    long long n = 4;
    std::vector<long long> bv(n, -1);
    std::vector<long long> bv_id(n, -1);
    long long bv_total = 0;

    filter_kernel(row_ptr, col_idx, C, bv, bv_id, &bv_total, n);

    EXPECT_GE(bv_total, 1);
    EXPECT_NE(bv_id[1], -1); 
}
