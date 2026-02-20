#include "ml/poly.h"
#include "ml/matrix.h"
#include <stdexcept>

namespace ml
{

    // Expand each feature x -> [x, x^2, ..., x^degree] (no cross terms)
    Mat poly_expand(const Mat &X, int degree)
    {
        if (degree < 1)
            throw std::runtime_error("poly_expand: degree must be >= 1");
        if (X.empty())
            return {};

        const size_t n = X.size();
        const size_t d = X[0].size();

        Mat Z;
        Z.reserve(n);

        for (size_t i = 0; i < n; i++)
        {
            if (X[i].size() != d)
                throw std::runtime_error("poly_expand: ragged X");

            VecN row;
            row.reserve(d * static_cast<size_t>(degree));

            for (size_t j = 0; j < d; j++)
            {
                double val = 1.0;
                for (int p = 1; p <= degree; p++)
                {
                    val *= X[i][j]; // val = x^p
                    row.push_back(val);
                }
            }
            Z.push_back(std::move(row));
        }

        return Z;
    }

    // Ridge normal equation: (XtX + lambda I) w = Xty
    VecN ridge_normal_equation(const Mat &X, const VecN &y, double lambda)
    {
        if (lambda < 0)
            throw std::runtime_error("ridge_normal_equation: lambda must be >= 0");
        if (X.empty())
            throw std::runtime_error("ridge_normal_equation: empty X");
        if (y.size() != X.size())
            throw std::runtime_error("ridge_normal_equation: y mismatch");

        Mat Xt = transpose(X);
        Mat XtX = matmul(Xt, X);

        for (size_t i = 0; i < XtX.size(); i++)
            XtX[i][i] += lambda;

        XtX[0][0] -= lambda; // remove regularization on bias

        // Xty
        Mat ycol(y.size(), VecN(1, 0.0));
        for (size_t i = 0; i < y.size(); i++)
            ycol[i][0] = y[i];
        Mat XtyMat = matmul(Xt, ycol);

        VecN Xty(XtyMat.size(), 0.0);
        for (size_t i = 0; i < Xty.size(); i++)
            Xty[i] = XtyMat[i][0];

        return solve_gauss(XtX, Xty);
    }

} // namespace ml
