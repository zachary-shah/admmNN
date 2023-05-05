# baADMM
Convex Reformulation of ANN ReLU problem with GPU Acceleration 

### Todo List

Before Midterm Report (4/15)
- [ ] Read Paper: [Efficient Global Optimization of Two-layer ReLU Networks: Quadratic-time
Algorithms and Adversarial Training]([https://arxiv.org/pdf/2201.01965.pdf)
- [ ] Implement Exact ADMM solver for Two-layer ReLU (Algo 3.1) - Zach *Due Sunday*
- [ ] Test against PyTorch on problem with known exact solution (MNIST subset with 100% accuracy easy and possible) - Zach *Due Sunday*
- [ ] Implement Approximate ADMM solver - Miria *Due Friday*
- [ ] Test solver - Miria *Due Friday*
- [ ] Test exact and approximate ADMM solvers against more baselines (CVX, scnn-fista, PyTorch) - Daniel? 
- [ ] Summarize results in writeup 

For Final Project 
- [ ] Accelerate solvers with GPUs 
- [ ] Scale problem to CIFAR-10 classification, test against baselines
- [ ] Test for other problems with structure, where numerical acceleration may be possible and present 
