#include "ml/stats.h"
#include <cmath>
#include <stdexcept>

namespace ml
{

    VecN column(const Mat &X, size_t j)
    {
        if (X.empty())
            return {};
        if (j >= X[0].size())
            throw std::runtime_error("column: index out of range");

        VecN col;
        col.reserve(X.size());
        for (const auto &row : X)
        {
            col.push_back(row[j]);
        }
        return col;
    }

    double mean(const VecN &x)
    {
        if (x.empty())
            throw std::runtime_error("mean: empty vector");
        double sum = 0.0;
        for (double v : x)
            sum += v;
        return sum / static_cast<double>(x.size());
    }

    // Population variance (divide by n)
    double variance(const VecN &x)
    {
        if (x.empty())
            throw std::runtime_error("variance: empty vector");
        double mu = mean(x);
        double sum = 0.0;
        for (double v : x)
        {
            double d = v - mu;
            sum += d * d;
        }
        return sum / static_cast<double>(x.size());
    }

    double stddev(const VecN &x)
    {
        return std::sqrt(variance(x));
    }

} // namespace ml
