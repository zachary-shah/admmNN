# baADMM
Convex Reformulation of ANN ReLU problem with GPU Acceleration 

### Todo List

Zach Todo List (Due Sunday)
- [x] Read Paper Sections 3 (up to 3.2.2): [Efficient Global Optimization of Two-layer ReLU Networks: Quadratic-time
Algorithms and Adversarial Training](https://arxiv.org/pdf/2201.01965.pdf)
- [x] Implement Algo 3.1 of Paper (Approximate ADMM solver)

Todo List (Due Wednesday Night)
- [x] Fix zach's bad baby optimizer to match results of baseline
- [ ] Read Paper Sections 3 (up to 3.2.2): [Efficient Global Optimization of Two-layer ReLU Networks: Quadratic-time
Algorithms and Adversarial Training](https://arxiv.org/pdf/2201.01965.pdf)
- [ ] Implement Baby SCNN to get baseline

Before Midterm Report (4/15)
- [ ] Test against PyTorch on problem with known exact solution (MNIST subset with 100% accuracy easy and possible) *Due Sunday*
- [ ] Improve time complexity of Ax = b
    - [ ] Try other pre-conditioners besides Cholesky for linear sys solving (Miria)
    - [ ] Conjugate Gradient steps to solve (Daniel) 
- [ ] Summarize results in midterm writeup 

For Final Project 
- [ ] GPU Acceleration
    - [ ] Hook up to JAX
    - [ ] Perf tests to demonstrate speedup 
- [ ] Test exact and approximate ADMM solvers against more baselines (CVX, scnn-fista, PyTorch)
- [ ] Scale problem to CIFAR-10 classification, test against baselines
- [ ] Test for other problems with structure, where numerical acceleration may be possible and present 
