#ifndef PRINTER_HPP
#define PRINTER_HPP

#include <string>
#include <vector>

void print_graph_info(const std::string &graph_file, const std::string &comm_file, long long n, long long m, long long total_community);
void print_vector(const std::vector<long long> &vec, const std::string &label);
void print_double_vector(const std::vector<double> &vec, const std::string &label);
void print_matrix(const std::vector<double> &matrix, long long rows, long long cols, const std::string &label);

#endif
