#include "ml/linreg.h"
#include "ml/matrix.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <execution>

namespace ml
{

    Mat add_bias(const Mat &X)
    {
        if (X.empty())
            return {};
        Mat Xb = X;
        for (auto &row : Xb)
            row.insert(row.begin(), 1.0);
        return Xb;
    }

    double mse(const Mat &X, const VecN &y, const VecN &w)
    {
        if (X.empty())
            throw std::runtime_error("mse: empty X");
        if (y.size() != X.size())
            throw std::runtime_error("mse: y size mismatch");
        VecN yhat = matvec(X, w);
        double s = 0.0;
        for (size_t i = 0; i < y.size(); i++)
        {
            double e = yhat[i] - y[i];
            s += e * e;
        }
        return s / static_cast<double>(y.size());
    }

    // ---------- helper: parallel gradient over features ----------
    static VecN grad_mse_par(const Mat &X,
                             const VecN &y,
                             const VecN &yhat)
    {
        const size_t n = X.size();
        const size_t d = X[0].size();

        VecN grad(d, 0.0);
        std::vector<size_t> cols(d);
        std::iota(cols.begin(), cols.end(), 0);

        std::for_each(std::execution::par, cols.begin(), cols.end(),
                      [&](size_t j)
                      {
                          double s = 0.0;
                          for (size_t i = 0; i < n; ++i)
                              s += X[i][j] * (yhat[i] - y[i]);
                          grad[j] = (2.0 / static_cast<double>(n)) * s;
                      });
        return grad;
    }

    // Normal equation: solve (XtX) w = Xty
    VecN normal_equation(const Mat &X, const VecN &y)
    {
        if (X.empty())
            throw std::runtime_error("normal_equation: empty X");
        if (y.size() != X.size())
            throw std::runtime_error("normal_equation: y mismatch");

        Mat Xt = transpose(X);

// perform parallelism if enabeled by user
#ifdef USE_PAR
        Mat XtX = matmul_par(Xt, X);

#else
        Mat XtX = matmul(Xt, X);
#endif

        // y as column matrix
        Mat ycol(y.size(), VecN(1, 0.0));
        for (size_t i = 0; i < y.size(); i++)
            ycol[i][0] = y[i];

// perform parallelism if enabeled by user
#ifdef USE_PAR
        Mat XtyMat = matmul_par(Xt, ycol);
#else
        Mat XtyMat = matmul(Xt, ycol);
#endif

        VecN Xty(XtyMat.size(), 0.0);
        for (size_t i = 0; i < Xty.size(); i++)
            Xty[i] = XtyMat[i][0];

        return solve_gauss(XtX, Xty);
    }

    VecN gradient_descent(const Mat &X, const VecN &y,
                          double alpha,
                          int iters,
                          double grad_tol = 1e-6,    // tolerance to break the loop at early convergence
                          int *iters_done = nullptr) // number to earlier iterations
    {
        if (X.empty())
            throw std::runtime_error("gradient_descent: empty X");
        if (y.size() != X.size())
            throw std::runtime_error("gradient_descent: y mismatch");
        if (alpha <= 0)
            throw std::runtime_error("gradient_descent: alpha must be positive");
        if (iters <= 0)
            throw std::runtime_error("gradient_descent: iters must be positive");

        const size_t n = X.size();
        const size_t d = X[0].size();
        VecN w(d, 0.0);

        for (int t = 0; t < iters; ++t)
        {

// perform parallelism if enabeled by user
#ifdef USE_PAR
            VecN yhat = matvec_par(X, w);         // parallel mat-vec
            VecN grad = grad_mse_par(X, y, yhat); // parallel gradient
#else
            VecN yhat = matvec(X, w); // serial fallback
            VecN grad(d, 0.0);
            const size_t n = X.size();
            for (size_t j = 0; j < d; ++j)
            {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i)
                    s += X[i][j] * (yhat[i] - y[i]);
                grad[j] = (2.0 / static_cast<double>(n)) * s;
            }
#endif
            // ---- convergence check
            double gnorm2 = 0.0;
            for (double g : grad)
                gnorm2 += g * g;
            // debugging log
            // if (t % 50 == 0) // print every 50 iters
            // std::printf("iter %d gradNorm = %.3e\n", t, std::sqrt(gnorm2));

            if (gnorm2 < grad_tol * grad_tol)
            {
                if (iters_done)
                    *iters_done = t + 1;
                break;
            }

            // ---- weight update
            for (size_t j = 0; j < d; ++j)
                w[j] -= alpha * grad[j];

            if (t == iters - 1 && iters_done) // reached max without converging
                *iters_done = iters;
        }
        return w;
    }

} // namespace ml
