# Genetic Programming Symbolic Regression

Repository for executing PySR-based tree-GP-like optimization on some benchmark datasets, plus several standard sklear-based ML models.

# Apptainer Container for GP-GOMEA and genetic-programming-sr

This repository provides a fully self-contained [Apptainer](https://apptainer.org) container for running [GP-GOMEA](https://github.com/lurovi/GP-GOMEA) and [genetic-programming-sr](https://github.com/lurovi/genetic-programming-sr) without Conda. It uses a minimal Ubuntu 20.04 base with specific compiler and Python library versions installed via `apt` and `pip`.

## 🔧 Container Features

- ✅ Ubuntu 20.04 base system
- ✅ C++ build environment with `gcc`, `cmake`, `boost`, `armadillo`, and `ninja`
- ✅ Python 3 with pinned versions of:
  - `numpy==1.24.4`
  - `pandas==2.0.3`
  - `scikit-learn==1.3.2`
  - `sympy==1.13.3`
  - and more
- ✅ Automatic build and installation of `pyGPGOMEA` (Python bindings)
- ✅ Clones and includes both:
  - `GP-GOMEA`
  - `genetic-programming-sr`
- ✅ Built without Conda for maximum portability

---

## 🚀 Build Instructions

Make sure you have Apptainer installed (formerly Singularity). Then build the container:

```bash
apptainer build gpsr.sif gpsr.def
