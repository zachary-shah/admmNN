# ADMM-based Neural Networks
Convex Reformulation of ANN ReLU problem with GPU Acceleration 

## Overview

We demonstrate a scalable reformulation of the non-convex training landscape of a ReLU-activated neural network as a convex optimization problem solved with variants of the Alternating Direction Method of Multipliers (ADMM) algorithm. This is a significant step towards achieving globally optimal interpretable results. We examine three practical ADMM based methods for solving this reformulated problem, and examine their performance with GPU acceleration on PyTorch and JAX. In order to meliorate the expensive primal step bottleneck of ADMM, we incorporate a randomized block-coordinate descent (RBCD) variant. We also experiment with NysADMM, which treats the primal update step as a linear solve with a randomized low-rank Nystrom approximation. This project examines the scalability and acceleration of these methods, in order to encourage applications across a wide range of statistical learning settings. Results show promising directions for scaling ADMM with accelerated GPU techniques to optimize two-layer neural networks.

## Further Reading

This work and relevant citations are summarized in a paper located in the ```testing``` folder.

## Use

An example of use of the optimizer is shown in ```tutorial_runner.ipynb```.

For a full description of the ```CReLU_MLP``` (Convex ReLU-Activated Multi-layer Perceptron) Optimization class, see ```relu_solver.py```.

## Supported Optimization Methods

This optimizer supports the following backends to train a 2-layer ReLU Network: 

- Numpy (cpu)
- Torch (cpu, gpu)
- JAX (cpu, gpu)

The primal update step of ADMM involves a large linear solve. By default, the optimizer will solve this system with a Cholseky decomposition, but this is unstable for high data dimensions. We support the following methods to approximate this linear solve for memory efficiency: 

- RBCD (randomized block coordinate descent)
- CG (Conjugate Gradient)
- PCG (Preconditioned Conjugate Gradient) with Conidioners:
    - Diagonal (Jacobi-PCG)
    - Nystrom (NysADMM)

