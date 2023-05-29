# baADMM
Convex Reformulation of ANN ReLU problem with GPU Acceleration 

### UPDATED REPO STRUCTURE 

Repo now has the following improvements implemented: 
- one ADMM solver for ADMM and xADMM-RBCD
- ADMM_Params object to abstract away parameters from end-state
- mnp library to provide backend flexibility for math operations
- typing hints for all functions


Things still to-do to validate repo: 
- [ ] Rewrite test_mnist.py to work with new runner config - Zach
- [ ] Finish docs - Zach
     - [ ] relu_solver.py docs
     - [ ] optimizers.py docs
     - [ ] 
- [ ] Validate numpy backend
- [ ] Validate torch backend - Miria
- [ ] Validate jax backend
    - [ ] Implement full jax backend (only added function wrappers) - Miria
- [ ] Implement ADMM CG - Daniel 
    - [ ] Add appropriate parameters for CG / PCG in utils.admm_utils.ADMM_Params
    - [ ] Implement CG / PCG in utils.primal_update_utils.ADMM_cg_update()

### Todo List

Zach Todo List (Due Sunday)
- [x] Read Paper Sections 3 (up to 3.2.2): [Efficient Global Optimization of Two-layer ReLU Networks: Quadratic-time
Algorithms and Adversarial Training](https://arxiv.org/pdf/2201.01965.pdf)
- [x] Implement Algo 3.1 of Paper (Approximate ADMM solver)

Todo List (Due Wednesday Night)
- [x] Fix zach's bad baby optimizer to match results of baseline
- [x] Read Paper Sections 3 (up to 3.2.2): [Efficient Global Optimization of Two-layer ReLU Networks: Quadratic-time
Algorithms and Adversarial Training](https://arxiv.org/pdf/2201.01965.pdf)
- [x] Implement Baby SCNN to get baseline

Before Midterm Report (4/15)
- [x] Test against PyTorch on problem with known exact solution (MNIST subset with 100% accuracy easy and possible) *Due Sunday*
- [ ] Improve time complexity of Ax = b
    - [ ] Try other pre-conditioners besides Cholesky for linear sys solving (Miria)
    - [ ] Conjugate Gradient steps to solve (Daniel) 
- [x] Summarize results in midterm writeup 

For Final Project 
- [ ] GPU Acceleration
    - [ ] Hook up to JAX
    - [ ] Perf tests to demonstrate speedup 
- [ ] Test exact and approximate ADMM solvers against more baselines (CVX, scnn-fista, PyTorch)
- [ ] Scale problem to CIFAR-10 classification, test against baselines
- [ ] Test for other problems with structure, where numerical acceleration may be possible and present 


