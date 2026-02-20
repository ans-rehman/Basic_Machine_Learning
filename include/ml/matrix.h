#pragma once
#include <vector>
#include <string>

namespace ml
{
    using VecN = std::vector<double>;
    using VecS = std::vector<std::string>;
    using Mat = std::vector<std::vector<double>>;

    // Basic ops
    Mat transpose(const Mat &A);
    Mat matmul(const Mat &A, const Mat &B);
    VecN matvec(const Mat &A, const VecN &x);
    double inner_prod_par(const VecN &, const VecN &);
    VecN matvec_par(const Mat &, const VecN &);
    Mat matmul_par(const Mat &, const Mat &);

    // Solvers
    VecN solve_gauss(Mat A, VecN b); // solves A w = b
}
