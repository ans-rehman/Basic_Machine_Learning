set datafile separator ','
set terminal pngcairo size 1000,600
set output 'results/weights_absdiff.png'
set title 'Absolute weight difference |w-NE - w-GD|'
set ylabel '|Δw|'
set style data histograms
set style fill solid 0.7 border -1
set boxwidth 0.6
set grid ytics
set xtics rotate by -15
plot 'results/weights_absdiff.csv' using 2:xtic(1) title 'abs diff'
