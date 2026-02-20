#pragma once
#include "matrix.h"

namespace ml
{

    // Expand each feature to powers 1..degree (simple polynomial features)
    Mat poly_expand(const Mat &X, int degree);

    // Ridge (L2) normal equation: (XtX + lambda I)w = Xty
    VecN ridge_normal_equation(const Mat &X, const VecN &y, double lambda);

}
