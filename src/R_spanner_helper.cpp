#include "R_spanner_helper.hpp"

long long read_args(long long argc, char **argv, std::string &filename, std::string &comm_filename, std::vector<long long> &target_communities)
{
    for (long long i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-g" && i + 1 < argc)
            filename = argv[++i];
        else if (arg == "-c" && i + 1 < argc)
            comm_filename = argv[++i];
        else if (arg == "-t" && i + 1 < argc)
            target_communities.push_back(atoi(argv[++i]));
        else
        {
            std::cerr << "Error: Unknown or incomplete flag '" << arg << "'\n";
            return 1;
        }
    }
    return 0;
}

void readCommunity(const std::string &filename, std::vector<long long> &C, const std::unordered_map<int, int> &comm_map)
{
    std::ifstream file(filename);
    std::string line;
    long long i = 0, c;

    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
        exit(1);
    }

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;

        std::istringstream iss(line);
        iss >> c;

        if (comm_map.find(c) == comm_map.end())
            C[i] = 0;
        else
            C[i] = comm_map.at(c);
        ++i;
    }

    file.close();
}

CSR mtxToCSR(const std::string &filename, long long *n, long long *m)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open graph file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line) && (line[0] == '%' || line[0] == '#'));

    std::istringstream dims(line);
    long long numRows, numCols, numEdges;
    dims >> numRows >> numCols >> numEdges;
    *n = numRows;
    *m = numEdges;

    std::vector<std::map<long long, long long>> rows(numRows);

    for (long long i = 0; i < numEdges; ++i)
    {
        std::getline(file, line);
        std::istringstream iss(line);
        long long row, col;
        iss >> row >> col;
        row--; col--;  // convert to 0-based

        rows[row][col] = 0;
        rows[col][row] = 0; // undirected graph
    }

    CSR csr;
    csr.row_ptr.push_back(0);
    for (const auto &row : rows)
    {
        for (const auto &kv : row)
        {
            csr.col_idx.push_back(kv.first);
            csr.data.push_back(kv.second);
        }
        csr.row_ptr.push_back(csr.col_idx.size());
    }

    return csr;
}
