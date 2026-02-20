#pragma once
#include <string>
#include <vector>

namespace ml
{

    // Write x-y series to CSV: columns are "x,y"
    void write_xy_csv(const std::string &path,
                      const std::vector<double> &x,
                      const std::vector<double> &y,
                      const std::string &x_name = "x",
                      const std::string &y_name = "y");

    // Convenience: write only y (x is 0..n-1)
    void write_y_csv(const std::string &path,
                     const std::vector<double> &y,
                     const std::string &y_name = "y");

    // Generate a gnuplot script that plots one CSV (x,y) into a PNG
    void write_gnuplot_script_xy_png(const std::string &script_path,
                                     const std::string &csv_path,
                                     const std::string &png_path,
                                     const std::string &title,
                                     const std::string &x_label,
                                     const std::string &y_label);

    void write_two_series_csv(const std::string &path,
                              const std::vector<double> &x,
                              const std::vector<double> &y1,
                              const std::vector<double> &y2,
                              const std::string &x_name,
                              const std::string &y1_name,
                              const std::string &y2_name);

    void write_gnuplot_script_two_series_png(const std::string &script_path,
                                             const std::string &csv_path,
                                             const std::string &png_path,
                                             const std::string &title,
                                             const std::string &x_label,
                                             const std::string &y_label,
                                             const std::string &s1_name,
                                             const std::string &s2_name);

    // Optional: run gnuplot script (returns true if command succeeded)
    bool run_gnuplot(const std::string &script_path);

    void write_gnuplot_script_bars_png(const std::string &script_path,
                                       const std::string &csv_path,
                                       const std::string &png_path,
                                       const std::string &title,
                                       const std::string &y_label,
                                       int value_col,                // which CSV column to plot (2=train_mse, 3=val_mse, 4=time_sec)
                                       const std::string &value_name // legend name
    );

} // namespace ml
