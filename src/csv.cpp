#include "ml/csv.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ml
{

    VecS split_csv(const std::string &line)
    {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }
        return row;
    }

    Mat load_numeric_matrix(const std::string &filename,
                            const std::vector<size_t> &skip_cols,
                            std::vector<std::string> *kept_headers)
    {
        std::ifstream fin(filename);
        if (!fin.is_open())
        {
            throw std::runtime_error("load_numeric_matrix: failed to open " + filename);
        }

        std::string headerLine;
        if (!std::getline(fin, headerLine))
        {
            throw std::runtime_error("load_numeric_matrix: empty file " + filename);
        }

        auto headers = split_csv(headerLine);

        // Precompute skip mask for O(1) checks
        std::vector<bool> skip(headers.size(), false);
        for (size_t idx : skip_cols)
        {
            if (idx < skip.size())
                skip[idx] = true;
        }

        // Fill kept headers (numeric columns only)
        if (kept_headers)
        {
            kept_headers->clear();
            for (size_t j = 0; j < headers.size(); j++)
            {
                if (!skip[j])
                    kept_headers->push_back(headers[j]);
            }
        }

        Mat X;
        std::string line;
        while (std::getline(fin, line))
        {
            if (line.empty())
                continue;

            auto row = split_csv(line);
            if (row.size() != headers.size())
            {
                // If you want strict mode, throw. For now, skip malformed lines.
                continue;
            }

            VecN numeric_row;
            numeric_row.reserve(headers.size() - skip_cols.size());

            for (size_t j = 0; j < row.size(); j++)
            {
                if (skip[j])
                    continue;
                numeric_row.push_back(std::stod(row[j]));
            }
            X.push_back(std::move(numeric_row));
        }

        return X;
    }

} // namespace ml
