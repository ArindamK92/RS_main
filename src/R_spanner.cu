#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <chrono>

#include "common.hpp"
#include "R_spanner_helper.hpp"
#include "printer.hpp"
#include "R_spanner_kernels.cuh"


int main(long long argc, char **argv)
{
    using namespace std::chrono;

    
    std::map<std::string, long long> timers;

    auto t0 = high_resolution_clock::now();
    std::string filename, comm_filename;
    long long max_comm_id = 0, total_community = 0;
    long long n, m, bv_total = 0;
    double maxWeight = 0.0;
    std::vector<long long> target_communities;

    read_args(argc, argv, filename, comm_filename, target_communities);

    std::unordered_map<int, int> comm_map;
    long long c_id = 1;
    for (auto c : target_communities) {
        comm_map[c] = c_id++;
        DEBUG_PRINT("comm_map[" << c << "] = " << comm_map[c]);
    }
    total_community = c_id;
    // timers["Parse Args & Map Communities"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    // t0 = high_resolution_clock::now();
    CSR csr = mtxToCSR(filename, &n, &m);
    std::vector<long long> C(n);
    readCommunity(comm_filename, C, comm_map);
    // timers["Load Graph & Community"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    print_graph_info(filename, comm_filename, n, m, total_community);

    // t0 = high_resolution_clock::now();  // AK: make it a kernel if we time it
    std::vector<long long> degree(n);
    for (long long i = 0; i < n; ++i)
        degree[i] = csr.row_ptr[i + 1] - csr.row_ptr[i];

    std::vector<long long> bv(n, -1), bv_id(n, -1);
    CUDA_CHECK(cudaSetDevice(0));
    // timers["Degree + Init"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();


    // **** Algorithm starts here ****
    // auto time_start = high_resolution_clock::now();
    t0 = high_resolution_clock::now();
    filter_kernel(csr.row_ptr, csr.col_idx, C, bv, bv_id, &bv_total, n);
    timers["Step 1: Filter Border Vertices"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();
    std::cout << "Total border vertices: " << bv_total << "\n";
    
    // Test
    // print_vector(bv, "bv");
    // print_vector(bv_id, "bv_id");

    t0 = high_resolution_clock::now();
    long long interaction_matrix_col_size = total_community;
    std::vector<double> comm_counts(bv_total * interaction_matrix_col_size, 0);
    std::vector<long long> T(bv_total, 0);
    find_ngbr_comm_kernel(csr.row_ptr, csr.col_idx, bv, C, comm_counts, T, bv_total, interaction_matrix_col_size);
    timers["Step 2A: Find Neighbor Community Counts"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    t0 = high_resolution_clock::now();
    compute_edge_weights(comm_counts, T, bv_total, interaction_matrix_col_size);
    timers["Step 2B: Compute Edge Weights"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    t0 = high_resolution_clock::now();
    normalize_comm_counts(comm_counts);
    timers["Step 2C: Normalize Edge Weights"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    t0 = high_resolution_clock::now();
    std::vector<long long> bv_pred_count(bv_total + 1, 0);
    find_bv_pred_count_kernel(csr.row_ptr, csr.col_idx, bv, bv_id, C, bv_pred_count, bv_total);
    timers["Step 2D: Find BV Predecessor Count"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    t0 = high_resolution_clock::now();
    std::vector<long long> row_ptr_Gb(bv_total + 1, 0);
    compute_row_ptr(bv_pred_count, row_ptr_Gb, bv_total + 1);
    timers["Step 2E: Compute G' Row Pointer"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    t0 = high_resolution_clock::now();
    long long total_edges_Gb = row_ptr_Gb.back();
    std::vector<long long> col_idx_Gb(total_edges_Gb, 0);
    fill_col_idx(csr.row_ptr, csr.col_idx, bv, bv_id, C, bv_total, row_ptr_Gb, col_idx_Gb);
    timers["Step 2F: Fill G' Column Indices"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    t0 = high_resolution_clock::now();
    std::vector<double> R(n, 0);
    compute_score_MergeIntersection(row_ptr_Gb, col_idx_Gb, comm_counts, C, bv, R, degree, bv_total, interaction_matrix_col_size);
    timers["Step 3: Compute R Scores"] = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();

    // auto time_end = high_resolution_clock::now();
    // long long total_time = duration_cast<milliseconds>(time_end - time_start).count();



    // Test: Prlong long R-scores 
    for (long long i = 0, count = 0; i < n && count < 10; ++i) {
        if (bv_id[i] >= 0) {
            std::cout << "R-score(" << i + 1 << ") = " << R[i] << "\n";
            ++count;
        }
    }

    

    std::cout << "\n======= TIMING REPORT (in ms) =======\n";
    long long total_time2 = 0;
    for (const auto &p : timers) {
        std::cout << p.first << ": " << p.second << " ms\n";
        total_time2 += p.second;
    }
    // std::cout << "Total Execution Time: " << total_time << " ms\n";
    std::cout << "Total Execution Time:: " << total_time2 << " ms\n";

    return 0;
}
