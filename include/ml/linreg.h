#pragma once
#include "matrix.h"

namespace ml
{

    // Normal equation solution (via Gaussian elimination, no inverse)
    VecN normal_equation(const Mat &X, const VecN &y);

    static VecN grad_mse_par(const Mat &X,
                             const VecN &y,
                             const VecN &yhat);
    // Gradient descent
    VecN gradient_descent(const Mat &X, const VecN &y,
                          double alpha, int iters, double grad_tol, int *iters_done);

    // Evaluation
    double mse(const Mat &X, const VecN &y, const VecN &w);

    // Add bias column (intercept)
    Mat add_bias(const Mat &X);

}
