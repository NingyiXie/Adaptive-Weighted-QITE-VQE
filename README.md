# Adaptive Weighted QITE-VQE Algorithm for Combinatorial Optimization Problems

This repository provides the implementation and experimental data for the paper "An Adaptive Weighted QITE-VQE Algorithm for Combinatorial Optimization Problems".

## AWQV: Adaptive Weighted QITE-VQE

AWQV integrates updates from compressed Quantum Imaginary Time Evolution (cQITE) into gradient-based optimization using a dynamic weighting scheme. The scheme initially favors cQITE updates to establish a strong initialization and gradually shifts toward the gradient component as the energy expectation converges.

## Repository Structure

- `qite_vqe.py`: Implementation of the algorithms AWQV, QITE, cQITE, and VQE
- `usage_example.ipynb`: A Jupyter notebook showing the usage of the provided algorithms
- `maxcut_problems.json`: Dataset of MaxCut problem instances used in our experiments
- `results/`: Directory containing experimental results
- `plot_results.ipynb`: Notebook for reproducing the plots from the paper

## Dependencies

### Main Requirements:
- tensorcircuit
- torch
- tensorflow
- networkx
- qiskit-optimization (For GW algorithm implementation)

### Visualization & Analysis:
- pandas 
- seaborn
- matplotlib
- jupyterlab


