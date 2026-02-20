set datafile separator ','
set terminal pngcairo size 1000,600
set output 'results/gd_loss.png'
set title 'Gradient Descent Convergence (Train MSE vs Iterations)'
set xlabel 'Iteration'
set ylabel 'Train MSE'
set grid
plot 'results/gd_loss.csv' using 1:2 with lines title 'Train MSE'
