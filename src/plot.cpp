#include "ml/plot.h"
#include <fstream>
#include <stdexcept>
#include <cstdlib> // std::system

namespace ml
{

    void write_xy_csv(const std::string &path,
                      const std::vector<double> &x,
                      const std::vector<double> &y,
                      const std::string &x_name,
                      const std::string &y_name)
    {
        if (x.size() != y.size())
            throw std::runtime_error("write_xy_csv: x and y must have same size");

        std::ofstream fout(path);
        if (!fout.is_open())
            throw std::runtime_error("write_xy_csv: failed to open " + path);

        fout << x_name << "," << y_name << "\n";
        for (size_t i = 0; i < x.size(); i++)
        {
            fout << x[i] << "," << y[i] << "\n";
        }
    }

    void write_y_csv(const std::string &path,
                     const std::vector<double> &y,
                     const std::string &y_name)
    {
        std::ofstream fout(path);
        if (!fout.is_open())
            throw std::runtime_error("write_y_csv: failed to open " + path);

        fout << "x," << y_name << "\n";
        for (size_t i = 0; i < y.size(); i++)
        {
            fout << i << "," << y[i] << "\n";
        }
    }

    void write_gnuplot_script_xy_png(const std::string &script_path,
                                     const std::string &csv_path,
                                     const std::string &png_path,
                                     const std::string &title,
                                     const std::string &x_label,
                                     const std::string &y_label)
    {
        std::ofstream gp(script_path);
        if (!gp.is_open())
            throw std::runtime_error("write_gnuplot_script_xy_png: failed to open " + script_path);

        // Keep it simple and portable
        gp << "set datafile separator ','\n";
        gp << "set terminal pngcairo size 1000,600\n";
        gp << "set output '" << png_path << "'\n";
        gp << "set title '" << title << "'\n";
        gp << "set xlabel '" << x_label << "'\n";
        gp << "set ylabel '" << y_label << "'\n";
        gp << "set grid\n";
        gp << "plot '" << csv_path << "' using 1:2 with lines title '" << y_label << "'\n";
    }

    // bool run_gnuplot(const std::string& script_path) {
    // #ifdef _WIN32
    //     std::string cmd = "gnuplot \"" + script_path + "\"";
    // #else
    //     std::string cmd = "gnuplot \"" + script_path + "\"";
    // #endif
    //     int code = std::system(cmd.c_str());
    //     return code == 0;
    // }

    void write_gnuplot_script_bars_png(const std::string &script_path,
                                       const std::string &csv_path,
                                       const std::string &png_path,
                                       const std::string &title,
                                       const std::string &y_label,
                                       int value_col,
                                       const std::string &value_name)
    {
        std::ofstream gp(script_path);
        if (!gp.is_open())
            throw std::runtime_error("write_gnuplot_script_bars_png: failed to open " + script_path);

        gp << "set datafile separator ','\n";
        gp << "set terminal pngcairo size 1000,600\n";
        gp << "set output '" << png_path << "'\n";
        gp << "set title '" << title << "'\n";
        gp << "set ylabel '" << y_label << "'\n";
        gp << "set style data histograms\n";
        gp << "set style fill solid 0.7 border -1\n";
        gp << "set boxwidth 0.6\n";
        gp << "set grid ytics\n";
        gp << "set xtics rotate by -15\n";
        gp << "plot '" << csv_path << "' using " << value_col << ":xtic(1) title '" << value_name << "'\n";
    }

    bool run_gnuplot(const std::string &script_path)
    {
        std::string cmd = "gnuplot \"" + script_path + "\"";
        int code = std::system(cmd.c_str());
        return code == 0;
    }

    void write_two_series_csv(const std::string &path,
                              const std::vector<double> &x,
                              const std::vector<double> &y1,
                              const std::vector<double> &y2,
                              const std::string &x_name,
                              const std::string &y1_name,
                              const std::string &y2_name)
    {
        if (x.size() != y1.size() || x.size() != y2.size())
            throw std::runtime_error("write_two_series_csv: size mismatch");

        std::ofstream fout(path);
        if (!fout.is_open())
            throw std::runtime_error("write_two_series_csv: failed to open " + path);

        fout << x_name << "," << y1_name << "," << y2_name << "\n";
        for (size_t i = 0; i < x.size(); i++)
            fout << x[i] << "," << y1[i] << "," << y2[i] << "\n";
    }

    void write_gnuplot_script_two_series_png(const std::string &script_path,
                                             const std::string &csv_path,
                                             const std::string &png_path,
                                             const std::string &title,
                                             const std::string &x_label,
                                             const std::string &y_label,
                                             const std::string &s1_name,
                                             const std::string &s2_name)
    {
        std::ofstream gp(script_path);
        if (!gp.is_open())
            throw std::runtime_error("write_gnuplot_script_two_series_png: failed to open " + script_path);

        gp << "set datafile separator ','\n";
        gp << "set terminal pngcairo size 1000,600\n";
        gp << "set output '" << png_path << "'\n";
        gp << "set title '" << title << "'\n";
        gp << "set xlabel '" << x_label << "'\n";
        gp << "set ylabel '" << y_label << "'\n";
        gp << "set grid\n";
        gp << "plot '" << csv_path << "' using 1:2 with lines title '" << s1_name
           << "', '" << csv_path << "' using 1:3 with lines title '" << s2_name << "'\n";
    }

} // namespace ml
