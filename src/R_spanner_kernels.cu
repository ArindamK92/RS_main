#include "R_spanner_kernels.cuh"
#include "common.hpp"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/extrema.h> 
#include <thrust/execution_policy.h>


#include "cuCompactor.cuh"


// // Filter kernel OPTION 1

// __global__ void filter_kernel_device(
//     const long long *row_ptr,
//     const long long *col_idx,
//     const long long *C,
//     long long *bv,
//     long long *bv_id,
//     long long *count,
//     long long n)
// {
//     long long i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n) return;

//     long long start_n = row_ptr[i];
//     long long stop_n = row_ptr[i + 1];
//     long long own_comm = C[i];
//     long long x = 0, y = 0, count_distinct = 0;

//     for (long long j = start_n; j < stop_n; ++j) {
//         long long ngbr = col_idx[j];
//         long long ngbr_comm = C[ngbr];
//         if (own_comm == 0) return;
//         x |= (1 << ngbr_comm);
//         if (x != y) {
//             count_distinct++;
//             y = x;
//         }
//         if (count_distinct >= 3) {
//             long long idx = atomicAdd(count, 1);
//             bv[idx] = i;
//             bv_id[i] = idx;
//             return;
//         }
//     }
// }

// // It filters out the community border vertices.
// /// An effective border vertex should have at least 3 communities (it may include comm 0 or own comm also) connected to it.
// /// If 2 or less communities are connected freq becomes 1 => H becomes 0 => edge weight becomes 0
// void filter_kernel(
//     const std::vector<long long> &row_ptr,
//     const std::vector<long long> &col_idx,
//     const std::vector<long long> &C,
//     std::vector<long long> &bv,
//     std::vector<long long> &bv_id,
//     long long *bv_total,
//     long long n)
// {
//     long long *d_row_ptr, *d_col_idx, *d_C, *d_bv, *d_bv_id, *d_count;
//     CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(long long) * row_ptr.size()));
//     CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(long long) * col_idx.size()));
//     CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
//     CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * bv.size()));
//     CUDA_CHECK(cudaMalloc(&d_bv_id, sizeof(long long) * bv_id.size()));
//     CUDA_CHECK(cudaMalloc(&d_count, sizeof(long long)));

//     CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(long long) * row_ptr.size(), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(long long) * col_idx.size(), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemset(d_bv, -1, sizeof(long long) * bv.size()));
//     CUDA_CHECK(cudaMemset(d_bv_id, -1, sizeof(long long) * bv_id.size()));
//     CUDA_CHECK(cudaMemset(d_count, 0, sizeof(long long)));

//     long long threadsPerBlock = THREADS_PER_BLOCK;
//     long long blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
//     filter_kernel_device<<<blocks, threadsPerBlock>>>(d_row_ptr, d_col_idx, d_C, d_bv, d_bv_id, d_count, n);
//     CUDA_CHECK(cudaGetLastError());

//     CUDA_CHECK(cudaMemcpy(bv.data(), d_bv, sizeof(long long) * bv.size(), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(bv_id.data(), d_bv_id, sizeof(long long) * bv_id.size(), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(bv_total, d_count, sizeof(long long), cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_row_ptr));
//     CUDA_CHECK(cudaFree(d_col_idx));
//     CUDA_CHECK(cudaFree(d_C));
//     CUDA_CHECK(cudaFree(d_bv));
//     CUDA_CHECK(cudaFree(d_bv_id));
//     CUDA_CHECK(cudaFree(d_count));
// }





// //// Filter kernel OPTION 2

// // Step 1: Mark border vertices
// __global__ void mark_border_vertices_kernel(
//     const long long *row_ptr,
//     const long long *col_idx,
//     const long long *C,
//     long long *is_bv,
//     long long n)
// {
//     long long i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n) return;

//     long long start_n = row_ptr[i];
//     long long stop_n = row_ptr[i + 1];
//     long long own_comm = C[i];
//     if (own_comm == 0) {
//         is_bv[i] = 0;
//         return;
//     }

//     long long x = 0, y = 0, count_distinct = 0;
//     for (long long j = start_n; j < stop_n; ++j) {
//         long long ngbr = col_idx[j];
//         long long ngbr_comm = C[ngbr];
//         x |= (1 << ngbr_comm);
//         if (x != y) {
//             count_distinct++;
//             y = x;
//         }
//         if (count_distinct >= 3) break;
//     }

//     is_bv[i] = (count_distinct >= 3) ? 1 : 0;
// }

// // Step 2: Scatter border vertex IDs into bv[], and map to bv_id[]
// __global__ void populate_bv_and_bv_id_kernel(
//     const long long *is_bv,
//     const long long *bv_offset,
//     long long *bv,
//     long long *bv_id,
//     long long n)
// {
//     long long i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n) return;

//     if (is_bv[i]) {
//         long long idx = bv_offset[i];
//         bv[idx] = i;
//         bv_id[i] = idx;
//     } else {
//         bv_id[i] = -1;
//     }
// }

// void filter_kernel(
//     const std::vector<long long> &row_ptr,
//     const std::vector<long long> &col_idx,
//     const std::vector<long long> &C,
//     std::vector<long long> &bv,
//     std::vector<long long> &bv_id,
//     long long *bv_total,
//     long long n)
// {
//     long long *d_row_ptr, *d_col_idx, *d_C, *d_bv, *d_bv_id;
//     CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(long long) * row_ptr.size()));
//     CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(long long) * col_idx.size()));
//     CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
//     CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * bv.size()));
//     CUDA_CHECK(cudaMalloc(&d_bv_id, sizeof(long long) * bv_id.size()));

//     CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(long long) * row_ptr.size(), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(long long) * col_idx.size(), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));

//     // Step 1: mark border vertices
//     thrust::device_vector<long long> d_is_bv(n);
//     thrust::device_vector<long long> d_bv_offset(n);

//     long long threadsPerBlock = THREADS_PER_BLOCK;
//     long long blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

//     mark_border_vertices_kernel<<<blocks, threadsPerBlock>>>(
//         d_row_ptr, d_col_idx, d_C, thrust::raw_pointer_cast(d_is_bv.data()), n);
//     CUDA_CHECK(cudaGetLastError());

//     // Step 2: exclusive scan on marks
//     thrust::exclusive_scan(thrust::device, d_is_bv.begin(), d_is_bv.end(), d_bv_offset.begin());

//     // Step 3: scatter valid indices into bv[] and update bv_id[]
//     populate_bv_and_bv_id_kernel<<<blocks, threadsPerBlock>>>(
//         thrust::raw_pointer_cast(d_is_bv.data()),
//         thrust::raw_pointer_cast(d_bv_offset.data()),
//         d_bv, d_bv_id, n);
//     CUDA_CHECK(cudaGetLastError());

//     // Compute bv_total
//     long long is_last = d_is_bv[n - 1];
//     long long offset_last = d_bv_offset[n - 1];
//     *bv_total = is_last + offset_last;

//     // Copy back results
//     CUDA_CHECK(cudaMemcpy(bv.data(), d_bv, sizeof(long long) * (*bv_total), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(bv_id.data(), d_bv_id, sizeof(long long) * bv_id.size(), cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_row_ptr));
//     CUDA_CHECK(cudaFree(d_col_idx));
//     CUDA_CHECK(cudaFree(d_C));
//     CUDA_CHECK(cudaFree(d_bv));
//     CUDA_CHECK(cudaFree(d_bv_id));
// }




// Filter kenrel OPTION 3

struct Vertex {
    long long Update;  // 1 if it's a border vertex, 0 otherwise
};

struct is_border_vertex {
    __host__ __device__
    long long operator()(long long val) const {
        return val != 0;
    }
};


__global__ void mark_border_vertices_kernel(
    const long long* row_ptr, const long long* col_idx, const long long* C,
    Vertex* flags, long long n)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < n; i += stride) {
        long long start = row_ptr[i];
        long long end = row_ptr[i + 1];
        long long own_comm = C[i];

        long long x = 0, y = 0, count_distinct = 0;
        for (long long j = start; j < end; ++j) {
            long long ngbr_comm = C[col_idx[j]];
            x |= (1 << ngbr_comm);
            if (x != y) {
                ++count_distinct;
                y = x;
            }
            if (count_distinct >= 3) break;
        }

        flags[i].Update = (count_distinct >= 3 && own_comm != 0) ? 1 : 0;
    }
}


__global__ void compute_bv_id_kernel(const long long* bv, long long* bv_id, long long bv_total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < bv_total; i += stride) {
        long long vertex_id = bv[i];
        bv_id[vertex_id] = i;
    }
}


void filter_kernel(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &C,
    std::vector<long long> &bv,
    std::vector<long long> &bv_id,
    long long *bv_total,
    long long n)
{
    // Device memory
    long long *d_row_ptr, *d_col_idx, *d_C;
    Vertex *d_flags;
    long long *d_bv, *d_bv_id;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(long long) * row_ptr.size()));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(long long) * col_idx.size()));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(Vertex) * n));
    CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * n));
    CUDA_CHECK(cudaMalloc(&d_bv_id, sizeof(long long) * n));
    CUDA_CHECK(cudaMemset(d_bv_id, -1, sizeof(long long) * n));  // default = -1

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(long long) * row_ptr.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(long long) * col_idx.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));

    long long threadsPerBlock = THREADS_PER_BLOCK;
    long long maxBlocks = 1024;  // safe limit for 1D grid on most GPUs
    long long blocks = std::min((n + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);

    // Mark valid border vertices
    mark_border_vertices_kernel<<<blocks, threadsPerBlock>>>(d_row_ptr, d_col_idx, d_C, d_flags, n);
    CUDA_CHECK(cudaGetLastError());

    // Compact valid vertex IDs
    long long compacted_length = cuCompactor::compact<Vertex, long long, is_border_vertex>(
        d_flags, d_bv, n, is_border_vertex(), threadsPerBlock);
    *bv_total = compacted_length;

    // Populate bv_id values
    long long bv_blocks = std::min((compacted_length + threadsPerBlock - 1) / threadsPerBlock, maxBlocks); // Limit to 1024 blocks
    compute_bv_id_kernel<<<bv_blocks, threadsPerBlock>>>(d_bv, d_bv_id, compacted_length);
    CUDA_CHECK(cudaGetLastError());

    bv.resize(compacted_length);
    CUDA_CHECK(cudaMemcpy(bv.data(), d_bv, sizeof(long long) * compacted_length, cudaMemcpyDeviceToHost));

    // Ensure bv_id has space; no need to reinitialize to -1
    if (bv_id.size() != static_cast<size_t>(n)) bv_id.resize(n);
    CUDA_CHECK(cudaMemcpy(bv_id.data(), d_bv_id, sizeof(long long) * n, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_bv_id));
}








__global__ void find_ngbr_comm_kernel_device(
    const long long *row_ptr,
    const long long *col_idx,
    const long long *bv,
    const long long *C,
    double *comm_counts,
    long long *T,
    long long bv_total,
    long long comm_count_col_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < bv_total; i += stride)
    {

    long long v = bv[i];
    long long start = row_ptr[v];
    long long stop = row_ptr[v + 1];
    long long offset = i * comm_count_col_size;
    long long total = 0;

    for (long long j = start; j < stop; ++j)
    {
        long long ngbr = col_idx[j];
        long long comm_id = C[ngbr];
        // if (comm_id == 0) continue; // we ignore comm id 0
        long long flat_idx = offset + comm_id;
        double prev_count = comm_counts[flat_idx];
        comm_counts[flat_idx] = prev_count + 1;
        total += 1;
    }
    T[i] = total;
    }
}

// It visits the neighbors of each border vertices and counts the number of neighbors from each community (count neighbor community kernel)
void find_ngbr_comm_kernel(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &bv,
    const std::vector<long long> &C,
    std::vector<double> &comm_counts,
    std::vector<long long> &T,
    long long bv_total,
    long long comm_count_col_size)
{
    long long *d_row_ptr, *d_col_idx, *d_bv, *d_C, *d_T;
    double *d_comm_counts;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(long long) * row_ptr.size()));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(long long) * col_idx.size()));
    CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * bv.size()));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
    CUDA_CHECK(cudaMalloc(&d_comm_counts, sizeof(double) * comm_counts.size()));
    CUDA_CHECK(cudaMalloc(&d_T, sizeof(long long) * T.size()));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(long long) * row_ptr.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(long long) * col_idx.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv, bv.data(), sizeof(long long) * bv.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_comm_counts, 0, sizeof(double) * comm_counts.size()));
    CUDA_CHECK(cudaMemset(d_T, 0, sizeof(long long) * T.size()));

    long long threadsPerBlock = THREADS_PER_BLOCK;
    long long maxBlocks = 1024;  // safe limit for 1D grid on most GPUs
    long long blocks = std::min((bv_total + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);
    find_ngbr_comm_kernel_device<<<blocks, threadsPerBlock>>>(
        d_row_ptr, d_col_idx, d_bv, d_C, d_comm_counts, d_T, bv_total, comm_count_col_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(comm_counts.data(), d_comm_counts, sizeof(double) * comm_counts.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(T.data(), d_T, sizeof(long long) * T.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_comm_counts));
    CUDA_CHECK(cudaFree(d_T));
}


__global__ void compute_edge_weights_kernel_device(
    double *comm_counts,
    long long *T,
    long long bv_total,
    long long comm_count_col_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < bv_total; i += stride)
    {
        long long offset = i * comm_count_col_size;
        long long last_idx = offset + comm_count_col_size - 1;

        double X_1 = 0.0, X_2 = 0.0;
        long long mod_L = 0;

        // Phase 1: Compute X_1, X_2, mod_L
        for (long long c = 0; c < comm_count_col_size; ++c) {
            long long flat_idx = offset + c;
            double f_c = comm_counts[flat_idx];
            if (f_c != 0.0) {
                X_1 += f_c * log2(f_c);
                X_2 += f_c;
                mod_L += 1;
            }
        }

        // Phase 2: Compute weights
        for (long long c = 0; c < comm_count_col_size; ++c) {
            long long flat_idx = offset + c;
            double f_c = comm_counts[flat_idx];
            double Y = T[i] - f_c; 

            double weight = 0.0;
            if (f_c != 0.0 && Y != 0.0) {
                weight = (-1.0) * (X_1 - log2(Y) * X_2 - f_c * log2(f_c) + f_c * log2(Y)) / Y * (mod_L - 1);
            }

            comm_counts[flat_idx] = (weight > 0.0) ? weight : 0.0;
        }
    }
}

// Compute edge weights for G'
void compute_edge_weights(std::vector<double> &comm_counts, std::vector<long long> &T, long long bv_total, long long comm_count_col_size)
{
    double *d_comm_counts;
    long long *d_T;
    size_t size = comm_counts.size() * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_comm_counts, size));
    CUDA_CHECK(cudaMalloc(&d_T, T.size() * sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(d_comm_counts, comm_counts.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, T.data(), T.size() * sizeof(long long), cudaMemcpyHostToDevice));

    long long threadsPerBlock = THREADS_PER_BLOCK;
    long long maxBlocks = 1024;  // safe limit for 1D grid on most GPUs
    long long blocks = std::min((bv_total + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);  

    compute_edge_weights_kernel_device<<<blocks, threadsPerBlock>>>(
        d_comm_counts, d_T, bv_total, comm_count_col_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(comm_counts.data(), d_comm_counts, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(T.data(), d_T, T.size() * sizeof(long long), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_comm_counts));
    CUDA_CHECK(cudaFree(d_T));
}


struct divide_by_max {
    double max_val;
    divide_by_max(double _max_val) : max_val(_max_val) {}
    __host__ __device__
    double operator()(const double& x) const {
        return x / max_val;
    }
};

// Find max weight and normalize
void normalize_comm_counts(std::vector<double>& comm_counts) {
    thrust::device_vector<double> comm_counts_dev(comm_counts.begin(), comm_counts.end());
    auto max_iter = thrust::max_element(comm_counts_dev.begin(), comm_counts_dev.end());
    double max_weight = *max_iter;
    std::cout << "max weight: " << max_weight << std::endl;
    thrust::transform(comm_counts_dev.begin(), comm_counts_dev.end(),
                      comm_counts_dev.begin(), divide_by_max(max_weight));  // Normalize all elements
    thrust::copy(comm_counts_dev.begin(), comm_counts_dev.end(), comm_counts.begin());
}



__global__ void find_bv_pred_count_kernel_device(
    const long long *row_ptr,
    const long long *col_idx,
    const long long *bv,
    const long long *bv_id,
    const long long *C,
    long long *bv_pred_count,
    long long bv_total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < bv_total; i += stride)
    {
        long long vertex_id = bv[i];
        long long start_n = row_ptr[vertex_id];
        long long stop_n = row_ptr[vertex_id + 1];
        long long own_comm = C[vertex_id];

        long long count = 0;
        for (long long j = start_n; j < stop_n; ++j) {
            long long ngbr = col_idx[j];
            if (bv_id[ngbr] >= 0 && C[ngbr] != own_comm) {
                count++;
            }
        }

        bv_pred_count[i] = count;
    }
}


// Find the max possible predecessor count for each border vertices. If a neighbor is also a bv it can be a predecessor in G'.
void find_bv_pred_count_kernel(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &bv,
    const std::vector<long long> &bv_id,
    const std::vector<long long> &C,
    std::vector<long long> &bv_pred_count,
    long long bv_total)
{
    long long *d_row_ptr, *d_col_idx, *d_bv, *d_bv_id, *d_C, *d_pred_count;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(long long) * row_ptr.size()));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(long long) * col_idx.size()));
    CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * bv.size()));
    CUDA_CHECK(cudaMalloc(&d_bv_id, sizeof(long long) * bv_id.size()));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
    CUDA_CHECK(cudaMalloc(&d_pred_count, sizeof(long long) * bv_pred_count.size()));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(long long) * row_ptr.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(long long) * col_idx.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv, bv.data(), sizeof(long long) * bv.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv_id, bv_id.data(), sizeof(long long) * bv_id.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_pred_count, 0, sizeof(long long) * bv_pred_count.size()));

    long long threadsPerBlock = THREADS_PER_BLOCK;
    long long maxBlocks = 1024;  // safe limit for 1D grid on most GPUs
    long long blocks = std::min((bv_total + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);
    find_bv_pred_count_kernel_device<<<blocks, threadsPerBlock>>>(
        d_row_ptr, d_col_idx, d_bv, d_bv_id, d_C, d_pred_count, bv_total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(bv_pred_count.data(), d_pred_count, sizeof(long long) * bv_pred_count.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_bv_id));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_pred_count));
}


void compute_row_ptr(
    const std::vector<long long> &in,
    std::vector<long long> &out,
    long long N)
{
    thrust::device_vector<long long> d_in(in.begin(), in.end());
    thrust::device_vector<long long> d_out(N);

    thrust::exclusive_scan(thrust::device, d_in.begin(), d_in.end(), d_out.begin());

    thrust::copy(d_out.begin(), d_out.end(), out.begin());
}

__global__ void fill_col_idx_kernel_device(
    const long long *row_ptr,
    const long long *col_idx,
    const long long *bv,
    const long long *bv_id,
    const long long *C,
    const long long *row_ptr_Gb,
    long long *col_idx_Gb,
    long long bv_total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < bv_total; i += stride)
    {
        long long v = bv[i];
        long long start = row_ptr[v];
        long long stop = row_ptr[v + 1];
        long long ptr = row_ptr_Gb[i];
        long long own_comm = C[v];

        for (long long j = start; j < stop; ++j)
        {
            long long ngbr = col_idx[j];
            long long ngbr_id = bv_id[ngbr];
            if (ngbr_id >= 0 && C[ngbr] != own_comm)
            {
                // a vertex has non-negative value (value is bv local id) if it is a bv
                // a neighbor in same community is not included as they creates Type-II triads and Type=II triads are handled seperately          
                col_idx_Gb[ptr] = ngbr_id;  // stores the bv local ids
                ptr = ptr + 1;
            }
        }
    }
}

void fill_col_idx(
    const std::vector<long long> &row_ptr,
    const std::vector<long long> &col_idx,
    const std::vector<long long> &bv,
    const std::vector<long long> &bv_id,
    const std::vector<long long> &C,
    long long bv_total,
    const std::vector<long long> &row_ptr_Gb,
    std::vector<long long> &col_idx_Gb)
{
    long long *d_row_ptr, *d_col_idx, *d_bv, *d_bv_id, *d_C, *d_row_ptr_Gb, *d_col_idx_Gb;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(long long) * row_ptr.size()));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(long long) * col_idx.size()));
    CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * bv.size()));
    CUDA_CHECK(cudaMalloc(&d_bv_id, sizeof(long long) * bv_id.size()));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_Gb, sizeof(long long) * row_ptr_Gb.size()));
    CUDA_CHECK(cudaMalloc(&d_col_idx_Gb, sizeof(long long) * col_idx_Gb.size()));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(long long) * row_ptr.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(long long) * col_idx.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv, bv.data(), sizeof(long long) * bv.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv_id, bv_id.data(), sizeof(long long) * bv_id.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr_Gb, row_ptr_Gb.data(), sizeof(long long) * row_ptr_Gb.size(), cudaMemcpyHostToDevice));

    long long threadsPerBlock = THREADS_PER_BLOCK;
    long long maxBlocks = 1024;  // safe limit for 1D grid on most GPUs
    long long blocks = std::min((bv_total + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);
    fill_col_idx_kernel_device<<<blocks, threadsPerBlock>>>(
        d_row_ptr, d_col_idx, d_bv, d_bv_id, d_C, d_row_ptr_Gb, d_col_idx_Gb, bv_total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(col_idx_Gb.data(), d_col_idx_Gb, sizeof(long long) * col_idx_Gb.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_bv_id));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_row_ptr_Gb));
    CUDA_CHECK(cudaFree(d_col_idx_Gb));
}


__global__ void compute_score_MergeIntersection_device(
    const long long *row_ptr_Gb,
    const long long *col_idx_Gb,
    const double *comm_counts,
    const long long *C,
    const long long *bv,
    double *R,
    const long long *degree,
    long long bv_total,
    long long comm_count_col_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;

    for (long long u = idx; u < bv_total; u += stride)
    {

        long long u_id = bv[u];
        long long u_c = C[u_id];
        long long start_nu = row_ptr_Gb[u];
        long long stop_nu = row_ptr_Gb[u + 1] - 1;

        double score = 0.0;

        for (long long j = start_nu; j <= stop_nu; ++j) {
            long long w = col_idx_Gb[j];
            long long w_id = bv[w];
            long long w_c = C[w_id];

            if (w_c == u_c) break;

            long long begin_nu = start_nu;
            long long end_nu = stop_nu;
            long long begin_nw = row_ptr_Gb[w];
            long long end_nw = row_ptr_Gb[w + 1] - 1;

            long long nu = col_idx_Gb[begin_nu];
            long long nw = col_idx_Gb[begin_nw];
            long long nu_id = bv[nu];
            long long nw_id = bv[nw];

            while (begin_nu <= end_nu && begin_nw <= end_nw) {
                if (C[nw_id] == u_c && nw_id != u_id) {
                    // Triad-II
                    long long vu_idx = nw * comm_count_col_size + u_c;
                    long long wu_idx = w * comm_count_col_size + u_c;
                    double W_vu = comm_counts[vu_idx];
                    double W_wu = comm_counts[wu_idx];
                    score += cbrt(W_vu * W_wu * W_wu);
                }
                else if (nw_id == nu_id && w_c != C[nw_id]) {
                    // Triad-I
                    long long vu_idx = nw * comm_count_col_size + u_c;
                    long long wu_idx = w * comm_count_col_size + u_c;
                    long long wv_idx = w * comm_count_col_size + C[nw_id];
                    double W_vu = comm_counts[vu_idx];
                    double W_wu = comm_counts[wu_idx];
                    double W_wv = comm_counts[wv_idx];
                    score += cbrt(W_vu * W_wu * W_wv);
                }

                bool comp1 = (nw_id >= nu_id);
                bool comp2 = (nw_id <= nu_id);
                bool nw_bound = (begin_nw == end_nw);
                bool nu_bound = (begin_nu == end_nu);

                if ((nw_bound && comp2) || (nu_bound && comp1)) break;

                if ((comp1 && !nu_bound) || nw_bound) {
                    ++begin_nu;
                    nu = col_idx_Gb[begin_nu];
                    nu_id = bv[nu];
                }
                if ((comp2 && !nw_bound) || nu_bound) {
                    ++begin_nw;
                    nw = col_idx_Gb[begin_nw];
                    nw_id = bv[nw];
                }
            }
        }

        long long deg = degree[u_id];
        if (deg > 1) {
            score /= (deg * (deg - 1));
            float scale = powf(10.0f, 4);
            score = round(score * scale) / scale;
            R[u_id] = score;
        }
    }
}

// ****Imp Assumption: It considers that the actual ids in the neighbor list of border vertices (in G') are stored in ascending order.****
/// It finds the intersection of neighbors using a merge technique
void compute_score_MergeIntersection(
    const std::vector<long long> &row_ptr_Gb,
    const std::vector<long long> &col_idx_Gb,
    const std::vector<double> &comm_counts,
    const std::vector<long long> &C,
    const std::vector<long long> &bv,
    std::vector<double> &R,
    const std::vector<long long> &degree,
    long long bv_total,
    long long comm_count_col_size)
{
    long long *d_row_ptr_Gb, *d_col_idx_Gb, *d_C, *d_bv, *d_degree;
    double *d_comm_counts, *d_R;

    CUDA_CHECK(cudaMalloc(&d_row_ptr_Gb, sizeof(long long) * row_ptr_Gb.size()));
    CUDA_CHECK(cudaMalloc(&d_col_idx_Gb, sizeof(long long) * col_idx_Gb.size()));
    CUDA_CHECK(cudaMalloc(&d_comm_counts, sizeof(double) * comm_counts.size()));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(long long) * C.size()));
    CUDA_CHECK(cudaMalloc(&d_bv, sizeof(long long) * bv.size()));
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(double) * R.size()));
    CUDA_CHECK(cudaMalloc(&d_degree, sizeof(long long) * degree.size()));

    CUDA_CHECK(cudaMemcpy(d_row_ptr_Gb, row_ptr_Gb.data(), sizeof(long long) * row_ptr_Gb.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx_Gb, col_idx_Gb.data(), sizeof(long long) * col_idx_Gb.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_comm_counts, comm_counts.data(), sizeof(double) * comm_counts.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), sizeof(long long) * C.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv, bv.data(), sizeof(long long) * bv.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_degree, degree.data(), sizeof(long long) * degree.size(), cudaMemcpyHostToDevice));

    long long threadsPerBlock = THREADS_PER_BLOCK;
    long long maxBlocks = 1024;  // safe limit for 1D grid on most GPUs
    long long blocks = std::min((bv_total + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);
    compute_score_MergeIntersection_device<<<blocks, threadsPerBlock>>>(
        d_row_ptr_Gb, d_col_idx_Gb, d_comm_counts, d_C, d_bv, d_R, d_degree,
        bv_total, comm_count_col_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(R.data(), d_R, sizeof(double) * R.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr_Gb));
    CUDA_CHECK(cudaFree(d_col_idx_Gb));
    CUDA_CHECK(cudaFree(d_comm_counts));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_degree));
}