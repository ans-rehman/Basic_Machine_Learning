#include "ml/csv.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace ml
{
    static std::string trim(const std::string& s)
    {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    static std::string strip_quotes(const std::string& s)
    {
        if (s.size() >= 2)
        {
            char q0 = s.front();
            char q1 = s.back();
            if ((q0 == '"' && q1 == '"') || (q0 == '\'' && q1 == '\''))
                return s.substr(1, s.size() - 2);
        }
        return s;
    }

    VecS split_csv(const std::string &line, char delim)
    {
        VecS row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delim))
        {
            cell = trim(cell);
            cell = strip_quotes(cell);
            row.push_back(cell);
        }
        return row;
    }

    Mat load_numeric_matrix(const std::string &filename,
                            const std::vector<size_t> &skip_cols,
                            std::vector<std::string> *kept_headers,
                            char delim,
                            const std::string& missing_token,
                            bool drop_rows_with_missing,
                            bool has_header,
                            const std::vector<std::string>* provided_headers)
    {
        std::ifstream fin(filename);
        if (!fin.is_open())
            throw std::runtime_error("load_numeric_matrix: failed to open " + filename);

        std::string line;

        VecS headers;

        if (has_header)
        {
            if (!std::getline(fin, line))
                throw std::runtime_error("load_numeric_matrix: empty file " + filename);

            headers = split_csv(line, delim);
        }
        else
        {
            // No header row in file (e.g., imports-85.data)
            if (provided_headers)
            {
                headers = *provided_headers;
            }
            else
            {
                // No headers provided. infer number of columns from first data row.
                std::streampos pos = fin.tellg();
                if (!std::getline(fin, line))
                    throw std::runtime_error("load_numeric_matrix: empty file " + filename);

                auto tmp = split_csv(line, delim);
                headers.resize(tmp.size());
                for (size_t i = 0; i < headers.size(); ++i)
                    headers[i] = "col" + std::to_string(i);

                // rewind to start of first data row
                fin.clear();
                fin.seekg(pos);
            }
        }

        // Precompute skip mask for O(1) checks
        std::vector<bool> skip(headers.size(), false);
        for (size_t idx : skip_cols)
        {
            if (idx < skip.size()) skip[idx] = true;
        }

        // Fill kept headers
        if (kept_headers)
        {
            kept_headers->clear();
            for (size_t j = 0; j < headers.size(); ++j)
            {
                if (!skip[j]) kept_headers->push_back(headers[j]);
            }
        }

        Mat X;

        while (std::getline(fin, line))
        {
            if (line.empty()) continue;

            auto row = split_csv(line, delim);

            // If row has fewer columns than expected, skip it
            if (row.size() != headers.size())
                continue;

            bool bad = false;
            VecN numeric_row;
            numeric_row.reserve(headers.size() - skip_cols.size());

            for (size_t j = 0; j < row.size(); ++j)
            {
                if (skip[j]) continue;

                if (!missing_token.empty() && row[j] == missing_token)
                {
                    if (drop_rows_with_missing)
                    {
                        bad = true;
                        break;
                    }
                    else
                    {
                        throw std::runtime_error("load_numeric_matrix: missing token in file: " + filename);
                    }
                }

                try
                {
                    numeric_row.push_back(std::stod(row[j]));
                }
                catch (...)
                {
                    bad = true;
                    break;
                }
            }

            if (!bad && !numeric_row.empty())
                X.push_back(std::move(numeric_row));
        }

        return X;
    }

} // namespace ml