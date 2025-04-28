#ifndef R_SPANNER_KERNELS_CUH
#define R_SPANNER_KERNELS_CUH

#include <vector>
// #include <cuda_runtime.h>

// Border vertex filtering
void filter_kernel(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &C,
    std::vector<long long> &bv,
    std::vector<long long> &bv_id,
    long long *bv_total,
    long long n
);

void find_ngbr_comm_kernel(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &bv,
    const std::vector<long long> &C,
    std::vector<double> &comm_counts,
    std::vector<long long> &T,
    long long bv_total,
    long long comm_count_col_size);


void compute_edge_weights(
    std::vector<double> &comm_counts,
    std::vector<long long> &T,
    long long bv_total,
    long long comm_count_col_size);

// double find_max_comm_count(const std::vector<double>& comm_counts);

void normalize_comm_counts(std::vector<double>& comm_counts);
    
void find_bv_pred_count_kernel(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &bv,
    const std::vector<long long> &bv_id,
    const std::vector<long long> &C,
    std::vector<long long> &bv_pred_count,
    long long bv_total);

void compute_row_ptr(
    const std::vector<long long> &in,
    std::vector<long long> &out,
    long long N);
    
void fill_col_idx(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &bv,
    const std::vector<long long> &bv_id,
    const std::vector<long long> &C,
    long long bv_total,
    const std::vector<long long> &row_ptr_Gb,
    std::vector<long long> &col_idx_Gb);

void compute_score_MergeIntersection(
    const std::vector<long long> &row_ptr_Gb,
    const std::vector<long long> &col_idx_Gb,
    const std::vector<double> &comm_counts,
    const std::vector<long long> &C,
    const std::vector<long long> &bv,
    std::vector<double> &R,
    const std::vector<long long> &degree,
    long long bv_total,
    long long comm_count_col_size);

#endif // R_SPANNER_KERNELS_CUH
