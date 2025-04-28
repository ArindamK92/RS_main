#include "printer.hpp"
#include <iostream>
#include <iomanip>

void print_graph_info(const std::string &graph_file, const std::string &comm_file, long long n, long long m, long long total_community)
{
    std::cout << "-------------------------------------\n";
    std::cout << "  Input Graph File: " << graph_file << "\n";
    std::cout << "  Community File:   " << comm_file << "\n";
    std::cout << "  Num Vertices:     " << n << "\n";
    std::cout << "  Num Edges:        " << m << "\n";
    std::cout << "  Communities Used: " << total_community - 1 << "\n";
    std::cout << "-------------------------------------\n";
}

void print_vector(const std::vector<long long> &vec, const std::string &label)
{
    std::cout << label << " [";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i != vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void print_double_vector(const std::vector<double> &vec, const std::string &label)
{
    std::cout << label << " [";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(2) << vec[i];
        if (i != vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void print_matrix(const std::vector<double> &matrix, long long rows, long long cols, const std::string &label)
{
    std::cout << label << ":\n";
    for (long long i = 0; i < rows; ++i)
    {
        for (long long j = 0; j < cols; ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}
