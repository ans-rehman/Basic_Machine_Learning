#include "ml/csv.h"
#include "ml/linreg.h"
#include "ml/poly.h"
#include "ml/plot.h"
#include "ml/stats.h"

#include <iostream>
#include <filesystem>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdio>

// Automobile headers for imports-85.data
static const ml::VecS headers_automobile = {
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
};

// -------------------- Dataset selection --------------------
enum class DatasetKind
{
    ForestFires,
    WineRed,
    WineWhite,
    AutoImports85
};

static DatasetKind parse_dataset_kind(int argc, char** argv)
{
    if (argc < 2) return DatasetKind::ForestFires;

    std::string s = argv[1];
    if (s == "forest")     return DatasetKind::ForestFires;
    if (s == "wine-red")   return DatasetKind::WineRed;
    if (s == "wine-white") return DatasetKind::WineWhite;
    if (s == "auto")       return DatasetKind::AutoImports85;

    std::cerr << "[WARN] Unknown dataset arg '" << s << "'. Using forestfires.\n";
    return DatasetKind::ForestFires;
}

// -------------------- Config --------------------
struct Config
{
    DatasetKind dataset = DatasetKind::ForestFires;
    std::string dataset_path = "datasets/forestfires.csv";

    double val_ratio = 0.2;
    unsigned seed = 42;

    int max_degree = 6;
    int ridge_degree = 4;

    // After standardization, larger alpha is stable
    double gd_alpha = 1e-2;
    int gd_iters = 3000;

    // GD early stopping threshold on ||grad||_2
    double grad_tol = 1e-4;

    // For logging GD curve only
    int gd_log_iters = 2000;

    std::vector<double> lambdas = {0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0};
};

// -------------------- Helpers --------------------
static void train_val_split(const ml::Mat &X, const ml::VecN &y,
                            double val_ratio,
                            ml::Mat &Xtr, ml::VecN &ytr,
                            ml::Mat &Xva, ml::VecN &yva,
                            unsigned seed)
{
    const size_t n = X.size();
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    size_t n_val = static_cast<size_t>(n * val_ratio);
    if (n_val < 1) n_val = 1;
    if (n_val >= n) n_val = n - 1;

    Xva.clear(); yva.clear();
    Xtr.clear(); ytr.clear();

    Xva.reserve(n_val); yva.reserve(n_val);
    Xtr.reserve(n - n_val); ytr.reserve(n - n_val);

    for (size_t k = 0; k < n; k++)
    {
        size_t i = idx[k];
        if (k < n_val) { Xva.push_back(X[i]); yva.push_back(y[i]); }
        else          { Xtr.push_back(X[i]); ytr.push_back(y[i]); }
    }
}

// Generic: last column is target; others are features
static void build_Xy_lastcol(const ml::Mat& all, ml::Mat& X, ml::VecN& y)
{
    if (all.empty()) throw std::runtime_error("build_Xy_lastcol: empty data");
    const size_t d = all[0].size();
    if (d < 2) throw std::runtime_error("build_Xy_lastcol: need >=2 cols");

    X.assign(all.size(), ml::VecN(d - 1));
    y.assign(all.size(), 0.0);

    for (size_t i = 0; i < all.size(); i++)
    {
        for (size_t j = 0; j + 1 < d; j++)
            X[i][j] = all[i][j];
        y[i] = all[i][d - 1];
    }
}

// Dataset summary: expects headers.size() == X[0].size()
static void summarize_dataset(const ml::Mat &X, const std::vector<std::string> &headers)
{
    if (X.empty())
    {
        std::cerr << "Dataset empty\n";
        return;
    }
    const size_t d = X[0].size();
    if (headers.size() != d)
    {
        std::cerr << "[WARN] summarize_dataset: headers (" << headers.size()
                  << ") != feature-dim (" << d << ")\n";
    }

    std::ofstream csv("results/column_summary.csv");
    csv << "feature,count,mean,variance,stddev\n";

    std::cout << "-------------- Dataset summary ------------\n";
    std::cout << std::fixed << std::setprecision(4);

    for (size_t j = 0; j < d; ++j)
    {
        ml::VecN col(X.size());
        for (size_t i = 0; i < X.size(); ++i) col[i] = X[i][j];

        double mu  = ml::mean(col);
        double var = ml::variance(col);
        double sd  = std::sqrt(var);

        std::string name = (j < headers.size() ? headers[j] : ("col" + std::to_string(j)));

        std::cout << std::setw(18) << name
                  << "  n=" << col.size()
                  << "  mean=" << mu
                  << "  var=" << var
                  << "  std=" << sd << "\n";

        csv << name << ','
            << col.size() << ','
            << mu << ','
            << var << ','
            << sd << '\n';
    }
    std::cout << "------------------------------------------\n";
}

static double eval_mse(const ml::Mat &X, const ml::VecN &y, const ml::VecN &w)
{
    return ml::mse(X, y, w);
}

static double l2_norm(const ml::VecN &w)
{
    double s = 0.0;
    for (double wi : w) s += wi * wi;
    return std::sqrt(s);
}

// -------------------- Standardization --------------------
struct Standardizer
{
    ml::VecN mean;
    ml::VecN stdev;
};

static Standardizer fit_standardizer(const ml::Mat &X)
{
    if (X.empty()) throw std::runtime_error("fit_standardizer: empty X");
    const size_t d = X[0].size();

    Standardizer s;
    s.mean.assign(d, 0.0);
    s.stdev.assign(d, 1.0);

    for (size_t j = 0; j < d; j++)
    {
        ml::VecN col(X.size());
        for (size_t i = 0; i < X.size(); ++i) col[i] = X[i][j];

        s.mean[j]  = ml::mean(col);
        s.stdev[j] = ml::stddev(col);
        if (s.stdev[j] < 1e-12) s.stdev[j] = 1.0; // avoid divide-by-zero
    }
    return s;
}

static ml::Mat transform_standardize(const ml::Mat &X, const Standardizer &s)
{
    ml::Mat Z = X;
    for (auto &row : Z)
        for (size_t j = 0; j < row.size(); j++)
            row[j] = (row[j] - s.mean[j]) / s.stdev[j];
    return Z;
}

// -------------------- Main --------------------
int main(int argc, char** argv)
{
    try
    {
#ifdef USE_PAR
        std::fprintf(stderr, "[INFO] Parallel STL build (USE_PAR) active\n");
#else
        std::fprintf(stderr, "[INFO] Serial build (USE_PAR not defined)\n");
#endif

        Config cfg;
        cfg.dataset = parse_dataset_kind(argc, argv);

        std::filesystem::create_directories("results");

// Load dataset (configure delimiter / skip-cols / header mode)
        ml::VecS kept_headers;
        ml::Mat all_numeric;

        char delim = ',';
        bool has_header = true;
        bool drop_rows_with_missing = false;
        std::string missing_token = "?";
        std::vector<size_t> skip_cols;
        const ml::VecS* provided_headers = nullptr;

        if (cfg.dataset == DatasetKind::ForestFires)
        {
            cfg.dataset_path = "datasets/forestfires.csv";
            delim = ',';
            has_header = true;
            skip_cols = {2, 3}; // month/day categorical
            drop_rows_with_missing = false;
        }
        else if (cfg.dataset == DatasetKind::WineRed)
        {
            cfg.dataset_path = "datasets/winequality-red.csv";
            delim = ';';
            has_header = true;
            skip_cols = {}; // all numeric
            drop_rows_with_missing = false;
        }
        else if (cfg.dataset == DatasetKind::WineWhite)
        {
            cfg.dataset_path = "datasets/winequality-white.csv";
            delim = ';';
            has_header = true;
            skip_cols = {}; // all numeric
            drop_rows_with_missing = false;
        }
        else if (cfg.dataset == DatasetKind::AutoImports85)
        {
            cfg.dataset_path = "datasets/imports-85.data";
            delim = ',';
            has_header = false;                 // .data has no header
            provided_headers = &headers_automobile;

            // numeric-only approach: skip categorical columns
            skip_cols = {
                2,  // make
                3,  // fuel-type
                4,  // aspiration
                5,  // num-of-doors
                6,  // body-style
                7,  // drive-wheels
                8,  // engine-location
                14, // engine-type
                15, // num-of-cylinders
                17  // fuel-system
            };

            // '?' appears in some numeric fields -> skip those rows
            drop_rows_with_missing = true;
        }

        all_numeric = ml::load_numeric_matrix(cfg.dataset_path,
                                              skip_cols,
                                              &kept_headers,
                                              delim,
                                              missing_token,
                                              drop_rows_with_missing,
                                              has_header,
                                              provided_headers);

        // Build X,y: target = last numeric column
        ml::Mat X_raw;
        ml::VecN y;
        build_Xy_lastcol(all_numeric, X_raw, y);

        // kept_headers includes target header in last position
        ml::VecS feature_headers = kept_headers;
        if (!feature_headers.empty())
            feature_headers.pop_back();

        summarize_dataset(X_raw, feature_headers);

// Train/val split + standardize
        ml::Mat Xtr_raw, Xva_raw;
        ml::VecN ytr, yva;
        train_val_split(X_raw, y, cfg.val_ratio, Xtr_raw, ytr, Xva_raw, yva, cfg.seed);

        {
            auto scaler = fit_standardizer(Xtr_raw);
            Xtr_raw = transform_standardize(Xtr_raw, scaler);
            Xva_raw = transform_standardize(Xva_raw, scaler);
        }

// (A) Linear Regression: Normal Eq vs GD (degree 1)
        std::cout << "\n**------------- Linear Regression Comparison ------------**\n";

        ml::Mat Xtr_lin = ml::add_bias(Xtr_raw);
        ml::Mat Xva_lin = ml::add_bias(Xva_raw);

        auto t1 = std::chrono::high_resolution_clock::now();
        ml::VecN w_ne = ml::normal_equation(Xtr_lin, ytr);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_ne = std::chrono::duration<double>(t2 - t1).count();

        int gd_iters_done = 0;
        auto t3 = std::chrono::high_resolution_clock::now();
        ml::VecN w_gd = ml::gradient_descent(Xtr_lin, ytr,
                                             cfg.gd_alpha, cfg.gd_iters,
                                             cfg.grad_tol, &gd_iters_done);
        auto t4 = std::chrono::high_resolution_clock::now();
        double time_gd = std::chrono::duration<double>(t4 - t3).count();

        double train_mse_ne = ml::mse(Xtr_lin, ytr, w_ne);
        double val_mse_ne   = ml::mse(Xva_lin, yva, w_ne);

        double train_mse_gd = ml::mse(Xtr_lin, ytr, w_gd);
        double val_mse_gd   = ml::mse(Xva_lin, yva, w_gd);

        std::cout << "\nNormal Equation:\n";
        std::cout << "Train MSE: " << train_mse_ne << "\n";
        std::cout << "Val MSE: "   << val_mse_ne   << "\n";
        std::cout << "Time (sec): " << time_ne << "\n";

        std::cout << "\nGradient Descent:\n";
        std::cout << "Train MSE: " << train_mse_gd << "\n";
        std::cout << "Val MSE: "   << val_mse_gd   << "\n";
        std::cout << "Time (sec): " << time_gd << "\n";
        std::cout << "Iterations: " << gd_iters_done << "\n";

        double diff = 0.0;
        for (size_t i = 0; i < w_ne.size(); i++)
        {
            double d = w_ne[i] - w_gd[i];
            diff += d * d;
        }
        std::cout << "\nWeight Difference (L2 norm): " << std::sqrt(diff) << "\n";

        // Save linear comparison CSV
        {
            std::ofstream comp("results/linear_comparison.csv");
            comp << "method,train_mse,val_mse,time_sec\n";
            comp << "Normal equation,"  << train_mse_ne << "," << val_mse_ne << "," << time_ne << "\n";
            comp << "Gradient descent," << train_mse_gd << "," << val_mse_gd << "," << time_gd << "\n";
        }

// (B) Plot: MSE vs polynomial degree
        std::vector<double> degrees, train_mse_deg, val_mse_deg;
        degrees.reserve(cfg.max_degree);
        train_mse_deg.reserve(cfg.max_degree);
        val_mse_deg.reserve(cfg.max_degree);

        for (int deg = 1; deg <= cfg.max_degree; deg++)
        {
            ml::Mat Xtr_poly = ml::add_bias(ml::poly_expand(Xtr_raw, deg));
            ml::Mat Xva_poly = ml::add_bias(ml::poly_expand(Xva_raw, deg));
            ml::VecN w = ml::normal_equation(Xtr_poly, ytr);

            degrees.push_back((double)deg);
            train_mse_deg.push_back(eval_mse(Xtr_poly, ytr, w));
            val_mse_deg.push_back(eval_mse(Xva_poly, yva, w));
        }

        ml::write_two_series_csv("results/mse_vs_degree.csv",
                                 degrees, train_mse_deg, val_mse_deg,
                                 "degree", "train_mse", "val_mse");

        ml::write_gnuplot_script_two_series_png("results/mse_vs_degree.gp",
                                                "results/mse_vs_degree.csv",
                                                "results/mse_vs_degree.png",
                                                "Train vs Validation MSE vs Polynomial Degree",
                                                "Polynomial Degree", "MSE",
                                                "Train", "Validation");

// (C) Ridge: error vs lambda + ||w|| vs lambda
        ml::Mat Xtr_r = ml::add_bias(ml::poly_expand(Xtr_raw, cfg.ridge_degree));
        ml::Mat Xva_r = ml::add_bias(ml::poly_expand(Xva_raw, cfg.ridge_degree));

        std::vector<double> train_mse_lam, val_mse_lam, wnorm_lam;
        train_mse_lam.reserve(cfg.lambdas.size());
        val_mse_lam.reserve(cfg.lambdas.size());
        wnorm_lam.reserve(cfg.lambdas.size());

        for (double lam : cfg.lambdas)
        {
            ml::VecN w = ml::ridge_normal_equation(Xtr_r, ytr, lam);
            train_mse_lam.push_back(eval_mse(Xtr_r, ytr, w));
            val_mse_lam.push_back(eval_mse(Xva_r, yva, w));
            wnorm_lam.push_back(l2_norm(w));
        }

        ml::write_two_series_csv("results/mse_vs_lambda.csv",
                                 cfg.lambdas, train_mse_lam, val_mse_lam,
                                 "lambda", "train_mse", "val_mse");

        ml::write_gnuplot_script_two_series_png("results/mse_vs_lambda.gp",
                                                "results/mse_vs_lambda.csv",
                                                "results/mse_vs_lambda.png",
                                                "Train vs Validation MSE vs L2 Regularization (Ridge)",
                                                "Lambda", "MSE",
                                                "Train", "Validation");

        ml::write_xy_csv("results/wnorm_vs_lambda.csv",
                         cfg.lambdas, wnorm_lam,
                         "lambda", "w_norm");

        ml::write_gnuplot_script_xy_png("results/wnorm_vs_lambda.gp",
                                        "results/wnorm_vs_lambda.csv",
                                        "results/wnorm_vs_lambda.png",
                                        "Weight Norm vs Lambda (Ridge)",
                                        "Lambda", "||w||_2");

// (D) GD convergence curve (loss vs iterations)
        ml::Mat Xtr_gd = ml::add_bias(Xtr_raw);

        ml::VecN w_log(Xtr_gd[0].size(), 0.0);
        std::vector<double> it_x, loss_y;
        it_x.reserve(cfg.gd_log_iters);
        loss_y.reserve(cfg.gd_log_iters);

        for (int t = 0; t < cfg.gd_log_iters; t++)
        {
            double L = ml::mse(Xtr_gd, ytr, w_log);
            it_x.push_back((double)t);
            loss_y.push_back(L);

            const size_t n = Xtr_gd.size();
            const size_t d = Xtr_gd[0].size();
            ml::VecN yhat = ml::matvec(Xtr_gd, w_log);

            ml::VecN grad(d, 0.0);
            for (size_t j = 0; j < d; j++)
            {
                double s = 0.0;
                for (size_t i = 0; i < n; i++)
                    s += Xtr_gd[i][j] * (yhat[i] - ytr[i]);
                grad[j] = (2.0 / (double)n) * s;
            }

            for (size_t j = 0; j < d; j++)
                w_log[j] -= cfg.gd_alpha * grad[j];
        }

        ml::write_xy_csv("results/gd_loss.csv", it_x, loss_y, "iter", "train_mse");

        ml::write_gnuplot_script_xy_png("results/gd_loss.gp",
                                        "results/gd_loss.csv",
                                        "results/gd_loss.png",
                                        "Gradient Descent Convergence (Train MSE vs Iterations)",
                                        "Iteration", "Train MSE");

// (E) Bar plots: Train MSE, Val MSE, Runtime
        ml::write_gnuplot_script_bars_png("results/linear_train_mse.gp",
                                          "results/linear_comparison.csv",
                                          "results/linear_train_mse.png",
                                          "Linear Regression: Train MSE (Normal Eq vs GD)",
                                          "Train MSE",
                                          2, "Train MSE");

        ml::write_gnuplot_script_bars_png("results/linear_val_mse.gp",
                                          "results/linear_comparison.csv",
                                          "results/linear_val_mse.png",
                                          "Linear Regression: Validation MSE (Normal Eq vs GD)",
                                          "Validation MSE",
                                          3, "Validation MSE");

        ml::write_gnuplot_script_bars_png("results/linear_time.gp",
                                          "results/linear_comparison.csv",
                                          "results/linear_time.png",
                                          "Linear Regression: Runtime (Normal Eq vs GD)",
                                          "Time (seconds)",
                                          4, "Runtime (sec)");

// Run gnuplot scripts
        bool ok_deg   = ml::run_gnuplot("results/mse_vs_degree.gp");
        bool ok_lam   = ml::run_gnuplot("results/mse_vs_lambda.gp");
        bool ok_wnorm = ml::run_gnuplot("results/wnorm_vs_lambda.gp");
        bool ok_gd    = ml::run_gnuplot("results/gd_loss.gp");
        bool ok_bar1  = ml::run_gnuplot("results/linear_train_mse.gp");
        bool ok_bar2  = ml::run_gnuplot("results/linear_val_mse.gp");
        bool ok_bar3  = ml::run_gnuplot("results/linear_time.gp");

        std::cout << "\n**-------------- Results Summary ---------------**\n";
        std::cout << "PNG plots:\n";
        std::cout << "  mse_vs_degree.png     (" << (ok_deg ? "OK" : "FAILED") << ")\n";
        std::cout << "  mse_vs_lambda.png     (" << (ok_lam ? "OK" : "FAILED") << ")\n";
        std::cout << "  wnorm_vs_lambda.png   (" << (ok_wnorm ? "OK" : "FAILED") << ")\n";
        std::cout << "  gd_loss.png           (" << (ok_gd ? "OK" : "FAILED") << ")\n";
        std::cout << "  linear_train_mse.png  (" << (ok_bar1 ? "OK" : "FAILED") << ")\n";
        std::cout << "  linear_val_mse.png    (" << (ok_bar2 ? "OK" : "FAILED") << ")\n";
        std::cout << "  linear_time.png       (" << (ok_bar3 ? "OK" : "FAILED") << ")\n";
        std::cout << "**---------------------------------------------**\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}