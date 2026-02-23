#pragma once
#include <string>
#include <vector>

namespace ml {

using VecN = std::vector<double>;
using Mat  = std::vector<VecN>;
using VecS = std::vector<std::string>;

VecS split_csv(const std::string& line, char delim = ',');

// If has_header=false, the first line is treated as data.
// If provided_headers != nullptr, those headers are used when has_header=false.
// If drop_rows_with_missing=true, any row with missing_token in a used column is skipped.
Mat load_numeric_matrix(const std::string& filename,
                        const std::vector<size_t>& skip_cols,
                        std::vector<std::string>* kept_headers,
                        char delim = ',',
                        const std::string& missing_token = "?",
                        bool drop_rows_with_missing = false,
                        bool has_header = true,
                        const std::vector<std::string>* provided_headers = nullptr);

} // namespace ml