#pragma once
#include <string>
#include <vector>
#include "matrix.h"

namespace ml
{

    // Split a CSV line by commas
    std::vector<std::string> split_csv(const std::string &line);

    // Load numeric matrix from forestfire-like CSV.
    // skip_cols: indices to skip (month/day)
    Mat load_numeric_matrix(const std::string &filename,
                            const std::vector<size_t> &skip_cols,
                            std::vector<std::string> *kept_headers = nullptr);

}
