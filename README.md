# Basic Machine Learning – C++17 from Scratch

A fully self-contained machine-learning pipeline written in **vanilla C++17**  
(no Eigen, Boost, Armadillo, etc.). It demonstrates:

* CSV parsing and descriptive statistics  
* Feature standardisation  
* Linear regression  
  * Closed-form (normal equation)  
  * Batch gradient descent (with early stopping)  
* Polynomial feature expansion  
* Ridge (L2) regularisation  
* C++17 parallel STL kernels (optional)  
* Automatic experiment scripts + Gnuplot figures  

All matrix/vector operations are built from nested `std::vector` ⟨double⟩.

---

## Directory layout

```text
.
├── CMakeLists.txt          ← build script
├── datasets/               ← original CSV (forestfires.csv)
├── include/ml/             ← public headers (matrix, linreg, poly, …)
├── src/                    ← implementation (.cpp)
├── results/                ← populated at runtime (CSV, .gp, .png)
└── report.pdf              ← experiment write-up (LaTeX source in /doc)
