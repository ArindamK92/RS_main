#include <gtest/gtest.h>
#include "R_spanner_kernels.cuh"

TEST(FindNgbrCommKernelTest, CountsPerCommunity)
{
    // Graph (directed):
    // Node 0 -> Node 1   (comm 1 -> 2)
    // Node 1 -> Node 0   (comm 2 -> 1)
    // Node 1 -> Node 2   (comm 2 -> 3)

    std::vector<long long> row_ptr = {0, 1, 3, 3};        
    std::vector<long long> col_idx = {1, 0, 2};           

    std::vector<long long> C = {1, 2, 3};                 
    std::vector<long long> bv = {0, 1};                   
    long long bv_total = bv.size();
    long long comm_count_col_size = 4;                    // num_communities + 1

    std::vector<double> comm_counts(bv_total * comm_count_col_size, 0.0);

    find_ngbr_comm_kernel(row_ptr, col_idx, bv, C, comm_counts, bv_total, comm_count_col_size);

    for (long long i = 0; i < bv_total; ++i) {
        std::cout << "bv[" << i << "] = ";
        for (long long j = 0; j < comm_count_col_size; ++j) {
            std::cout << comm_counts[i * comm_count_col_size + j] << " ";
        }
        std::cout << std::endl;
    }

    // Validate bv[0] = node 0 -> neighbor = node 1 (community 2)
    EXPECT_DOUBLE_EQ(comm_counts[0 * comm_count_col_size + 2], 1.0);
    EXPECT_DOUBLE_EQ(comm_counts[0 * comm_count_col_size + 3], 1.0);

    // Validate bv[1] = node 1 -> neighbors = node 0 (comm 1) and node 2 (comm 3)
    EXPECT_DOUBLE_EQ(comm_counts[1 * comm_count_col_size + 1], 1.0);  
    EXPECT_DOUBLE_EQ(comm_counts[1 * comm_count_col_size + 3], 2.0); 
}
