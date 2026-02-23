set datafile separator ','
set terminal pngcairo size 1000,600
set output 'results/wnorm_vs_lambda.png'
set title 'Weight Norm vs Lambda (Ridge)'
set xlabel 'Lambda'
set ylabel '||w||_2'
set grid
plot 'results/wnorm_vs_lambda.csv' using 1:2 with lines title '||w||_2'
