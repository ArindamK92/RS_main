#ifndef R_SPANNER_HELPER_HPP
#define R_SPANNER_HELPER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <map>

struct CSR {
    std::vector<long long> data;    // edge weights (can be 0)
    std::vector<long long> col_idx; // neighbor vertex indices
    std::vector<long long> row_ptr; // row pointer for CSR format
};

// Command-line argument parsing
long long read_args(long long argc, char **argv, std::string &filename, std::string &comm_filename, std::vector<long long> &target_communities);

// Reads community data from file
void readCommunity(const std::string &filename, std::vector<long long> &C, const std::unordered_map<int, int> &comm_map);

// Reads .mtx format graph and converts to CSR
CSR mtxToCSR(const std::string &filename, long long *n, long long *m);

#endif
