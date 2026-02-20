#include "ml/matrix.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <execution>
#include <numeric>

namespace ml
{

    Mat transpose(const Mat &A)
    {
        if (A.empty())
            return {};
        size_t m = A.size(), n = A[0].size();
        Mat T(n, VecN(m, 0.0));
        for (size_t i = 0; i < m; i++)
        {
            if (A[i].size() != n)
                throw std::runtime_error("transpose: ragged matrix");
            for (size_t j = 0; j < n; j++)
            {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }

    Mat matmul(const Mat &A, const Mat &B)
    {
        if (A.empty() || B.empty())
            return {};
        size_t m = A.size();
        size_t n = A[0].size();
        size_t n2 = B.size();
        size_t p = B[0].size();
        if (n != n2)
            throw std::runtime_error("matmul: dim mismatch");

        for (size_t i = 0; i < m; i++)
            if (A[i].size() != n)
                throw std::runtime_error("matmul: ragged A");
        for (size_t i = 0; i < n2; i++)
            if (B[i].size() != p)
                throw std::runtime_error("matmul: ragged B");

        Mat C(m, VecN(p, 0.0));

        for (size_t i = 0; i < m; i++)
        {
            for (size_t k = 0; k < n; k++)
            {
                double aik = A[i][k];
                for (size_t j = 0; j < p; j++)
                {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
        return C;
    }

    VecN matvec(const Mat &A, const VecN &x)
    {
        if (A.empty())
            return {};
        size_t m = A.size();
        size_t n = A[0].size();
        if (x.size() != n)
            throw std::runtime_error("matvec: dim mismatch");

        for (size_t i = 0; i < m; i++)
            if (A[i].size() != n)
                throw std::runtime_error("matvec: ragged A");

        VecN y(m, 0.0);
        for (size_t i = 0; i < m; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++)
                sum += A[i][j] * x[j];
            y[i] = sum;
        }
        return y;
    }

    // ===========================================================
    //  Parallel helpers  (require -DUSE_PAR to be used by user)
    // ===========================================================

    double inner_prod_par(const VecN &a, const VecN &b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("inner_prod_par: size mismatch");

        return std::transform_reduce(std::execution::par_unseq,
                                     a.begin(), a.end(),
                                     b.begin(),
                                     0.0);
    }

    VecN matvec_par(const Mat &A, const VecN &x)
    {
        const size_t m = A.size(), n = A[0].size();
        if (x.size() != n)
            throw std::runtime_error("matvec_par: dim mismatch");

        VecN y(m);

        std::vector<size_t> rows(m);
        std::iota(rows.begin(), rows.end(), 0);

        std::for_each(std::execution::par, rows.begin(), rows.end(),
                      [&](size_t i)
                      { y[i] = inner_prod_par(A[i], x); });

        return y;
    }

    Mat matmul_par(const Mat &A, const Mat &B)
    {
        const size_t m = A.size();
        const size_t n = A[0].size();
        const size_t n2 = B.size();
        const size_t p = B[0].size();
        if (n != n2)
            throw std::runtime_error("matmul_par: dim mismatch");

        Mat C(m, VecN(p, 0.0));

        std::vector<size_t> rows(m);
        std::iota(rows.begin(), rows.end(), 0);

        // Each thread owns one row of C → no data races
        std::for_each(std::execution::par, rows.begin(), rows.end(),
                      [&](size_t i)
                      {
                          for (size_t k = 0; k < n; ++k)
                          {
                              double aik = A[i][k];
                              for (size_t j = 0; j < p; ++j)
                                  C[i][j] += aik * B[k][j];
                          }
                      });
        return C;
    }

    // Solves A w = b via Gaussian elimination with partial pivoting.
    // Input A, b are copied (passed by value in header), so caller stays unchanged.
    VecN solve_gauss(Mat A, VecN b)
    {
        const size_t n = A.size();
        if (n == 0)
            throw std::runtime_error("solve_gauss: empty A");
        if (A[0].size() != n)
            throw std::runtime_error("solve_gauss: A not square");
        if (b.size() != n)
            throw std::runtime_error("solve_gauss: dim mismatch");

        for (size_t i = 0; i < n; i++)
            if (A[i].size() != n)
                throw std::runtime_error("solve_gauss: ragged A");

        const double EPS = 1e-12;

        // Forward elimination
        for (size_t col = 0; col < n; col++)
        {
            // Pivot
            size_t piv = col;
            double best = std::fabs(A[col][col]);
            for (size_t r = col + 1; r < n; r++)
            {
                double v = std::fabs(A[r][col]);
                if (v > best)
                {
                    best = v;
                    piv = r;
                }
            }
            if (best < EPS)
                throw std::runtime_error("solve_gauss: singular/ill-conditioned matrix");

            if (piv != col)
            {
                std::swap(A[piv], A[col]);
                std::swap(b[piv], b[col]);
            }

            // Eliminate rows below
            for (size_t r = col + 1; r < n; r++)
            {
                double factor = A[r][col] / A[col][col];
                A[r][col] = 0.0;
                for (size_t c = col + 1; c < n; c++)
                    A[r][c] -= factor * A[col][c];
                b[r] -= factor * b[col];
            }
        }

        // Back substitution
        VecN w(n, 0.0);
        for (int i = static_cast<int>(n) - 1; i >= 0; i--)
        {
            double sum = b[i];
            for (size_t j = i + 1; j < n; j++)
                sum -= A[i][j] * w[j];
            w[i] = sum / A[i][i];
        }
        return w;
    }

} // namespace ml
