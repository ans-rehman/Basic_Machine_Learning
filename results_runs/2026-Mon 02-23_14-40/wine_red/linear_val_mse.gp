set datafile separator ','
set terminal pngcairo size 1000,600
set output 'results/linear_val_mse.png'
set title 'Linear Regression: Validation MSE (Normal Eq vs GD)'
set ylabel 'Validation MSE'
set style data histograms
set style fill solid 0.7 border -1
set boxwidth 0.6
set grid ytics
set xtics rotate by -15
plot 'results/linear_comparison.csv' using 3:xtic(1) title 'Validation MSE'
