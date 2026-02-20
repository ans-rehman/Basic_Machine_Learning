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
```

## Optional: enable C++17 parallel STL

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_PAR=ON
cmake --build build -j
```

With ENABLE_PAR=ON the compiler is given -DUSE_PAR and the code paths
using std::execution::par / par_unseq are compiled in.
For this small dataset the overhead usually dominates; on larger
matrices you can see a 3–6 × speed-up.


### Dependencies

* C++17 compiler (GCC 10+, Clang 12+, MSVC 2019)
* CMake ≥ 3.21
* Optional gnuplot ≥ 5.4 for automatic plotting

Everything else is standard-library-only.
