set datafile separator ','
set terminal pngcairo size 1000,600
set output 'results/mse_vs_degree.png'
set title 'Train vs Validation MSE vs Polynomial Degree'
set xlabel 'Polynomial Degree'
set ylabel 'MSE'
set grid
plot 'results/mse_vs_degree.csv' using 1:2 with lines title 'Train', 'results/mse_vs_degree.csv' using 1:3 with lines title 'Validation'
