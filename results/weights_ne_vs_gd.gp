set datafile separator ','
set terminal pngcairo size 1000,600
set output 'results/weights_ne_vs_gd.png'
set title 'Weights: Normal Eq vs Gradient Descent'
set xlabel 'Weight index'
set ylabel 'Value'
set grid
plot 'results/weights_ne_vs_gd.csv' using 1:2 with lines title 'w-NE', 'results/weights_ne_vs_gd.csv' using 1:3 with lines title 'w-GD'
