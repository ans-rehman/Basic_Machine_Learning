#pragma once
#include "matrix.h"

namespace ml
{
    double mean(const VecN &x);
    double variance(const VecN &x); // population variance
    double stddev(const VecN &x);

    // Helpers
    VecN column(const Mat &X, size_t j);
}
