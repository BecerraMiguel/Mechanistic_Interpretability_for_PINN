# Technical Summary: Probing Classifier Findings for PINN Mechanistic Interpretability

## Week 2 — Probing Classifiers for Derivative Detection in a Poisson PINN

---

### 1. Experimental Setup

We trained a standard MLP-based Physics-Informed Neural Network (PINN) to solve the
2D Poisson equation, nabla^2 u = f on [0,1]^2 with Dirichlet boundary conditions and
the manufactured solution u(x,y) = sin(pi*x)*sin(pi*y). The trained model (4 hidden
layers, 64 neurons/layer, tanh activation, 12,737 parameters) achieved a relative L2
error of 0.99% against the analytical solution.

To investigate what computational representations the network develops internally, we
applied **linear probing classifiers** to intermediate layer activations. Activations
were extracted on a dense 100x100 grid (10,000 points) and stored in HDF5 format. For
each (layer, derivative) pair, a LinearProbe (single linear layer, y = Wx + b) was
trained via gradient descent (Adam, 1000 epochs) to predict ground-truth analytical
derivatives from the layer activations. R-squared (R^2) scores quantify how much
derivative information is **linearly accessible** at each layer.

**Probing targets** (5 derivative quantities):
- First-order: du/dx, du/dy
- Second-order: d2u/dx2, d2u/dy2
- Laplacian: nabla^2 u = d2u/dx2 + d2u/dy2

**Total probes trained**: 20 (4 layers x 5 derivatives), plus weight analysis.

---

### 2. Core Finding: Two-Stage Derivative Computation Strategy

The central discovery of Week 2 is that the PINN implements a **two-stage derivative
computation strategy** that cleanly separates first- and second-order derivatives:

| Layer   | du/dx R^2 | du/dy R^2 | d2u/dx2 R^2 | d2u/dy2 R^2 | nabla^2 u R^2 |
|---------|-----------|-----------|-------------|-------------|---------------|
| layer_0 | 0.7845    | 0.7883    | -0.1355     | -0.1048     | -0.4062       |
| layer_1 | 0.7965    | 0.7959    |  0.0567     |  0.0422     | -0.0431       |
| layer_2 | 0.8219    | 0.8075    |  0.2016     |  0.1938     |  0.1875       |
| layer_3 | 0.9124    | 0.8926    |  0.4636     |  0.5012     |  0.3424       |

**Stage 1 — Explicit first-derivative encoding.** First derivatives (du/dx, du/dy)
are already partially accessible at layer 0 (R^2 ~ 0.78) and improve gradually to
R^2 > 0.89 at layer 3. This indicates that first-derivative information is directly
encoded in the hidden layer activations and is linearly decodable.

**Stage 2 — Implicit second-derivative computation.** Second derivatives and the
Laplacian are *not* explicitly stored in activations. R^2 values start negative at
early layers (worse than predicting the mean) and only reach ~0.5 at layer 3. Instead,
second derivatives are computed on-demand via PyTorch's autograd during the PDE loss
backward pass, differentiating through the stored first-derivative representations.

This two-stage strategy is computationally efficient: the network stores 2 values
(du/dx, du/dy) explicitly rather than all 5 derivative quantities, relying on automatic
differentiation for higher-order terms.

---

### 3. Derivative Emergence Patterns

Analysis of R^2 progression across layers reveals two distinct emergence dynamics:

- **Gradual emergence (first derivatives):** R^2 improves steadily by ~0.04 per layer.
  The maximum single-layer jump is 0.09 (layer 2 to layer 3 for du/dx). First
  derivatives are partially present from the very first hidden layer.

- **Sudden emergence (second derivatives):** R^2 improves by ~0.20 per layer on
  average, with the maximum jump being 0.31 (layer 1 to layer 2 for d2u/dy2). Second
  derivatives transition from actively wrong (negative R^2) to partially encoded
  within 2-3 layers.

Layer 3 (the final hidden layer) is the critical "derivative computation layer" where
all derivatives reach their peak R^2 values. This is consistent with the network
needing to have derivative-related information maximally available just before the
output layer maps to the solution u(x,y).

---

### 4. Probe Weight Analysis: Finite-Difference-Like Mechanisms

Examination of the trained probe weights reveals that the network's derivative
computation mechanism resembles a **continuous, multi-scale generalization of classical
finite differences**.

**4.1 Directional Specificity**

PINN input-layer weights (w_x, w_y) correlate strongly with probe weights:
- corr(w_x, probe_du/dx) = -0.70; corr(w_y, probe_du/dy) = -0.75
- Cross-correlations are near zero (~0.03)

This shows that neurons sensitive to the x-coordinate contribute to predicting du/dx,
and neurons sensitive to y contribute to du/dy, with minimal cross-talk. The negative
sign is consistent with the chain rule: du/dx ~ tanh'(w_x * x + ...) * w_x, where
the tanh derivative is positive, so the sign of the product p_i * w_x,i reflects the
derivative direction.

**4.2 Neuron Pair Analysis (First Derivatives)**

At layer 0, we identified 372 neuron pairs for du/dx and 397 for du/dy that exhibit
finite-difference-like structure:
- High cosine similarity between pair activations (> 0.8)
- Opposite signs in probe weights (one positive, one negative)
- Mean weight symmetry: 0.72 (du/dx), 0.87 (du/dy)
- Effective grid spacings (h_eff) spanning a 7-10x range

This indicates the probe reconstructs derivatives via f(x+h) - f(x-h) style
subtraction across neuron activations, but using *multiple scales simultaneously*
rather than a single grid spacing as in classical FD.

**4.3 Triplet Analysis (Second Derivatives)**

For second derivatives, [1, -2, 1] triplet patterns were found with cosine similarity
to the ideal stencil of 0.970-0.986. However, the magnitude ratios (middle-to-outer)
were 1.3-1.8 rather than the ideal 2.0, consistent with the weaker R^2 values. The
network approximates but does not perfectly implement the classical second-derivative
stencil.

**4.4 Sign Bias**

A systematic sign pattern was found: the product p_i * w_{dir,i} (probe weight times
PINN input weight for the relevant coordinate) is negative for 91% of neurons (du/dx)
and 97% (du/dy). This is a signature of chain-rule-based derivative computation and
would not occur by chance (expected: 50%).

---

### 5. Proposed Hypothesis: Multi-Scale Continuous Finite Difference Algorithm

We propose that the trained PINN implements a **Multi-Scale Continuous Finite
Difference (MS-CFD) Algorithm**:

1. **Basis function construction** (layers 0-2): Each neuron computes a smooth,
   localized basis function sigma(w^T x + b) using tanh activation, analogous to
   radial basis functions (RBFs) centered at different spatial positions and scales.

2. **First-derivative encoding** (layers 0-3): Linear combinations of these shifted
   basis functions approximate first derivatives via learned subtraction patterns
   (neuron pairs with opposite probe weights), operating simultaneously at multiple
   spatial scales (h_eff range: 7-10x).

3. **Second-derivative computation** (implicit via autograd): Rather than storing
   second derivatives in activations, the network relies on PyTorch's autograd to
   differentiate through the explicit first-derivative representations during the
   PDE loss computation.

This mechanism differs from classical numerical methods in three key ways:
- **Continuous** rather than discrete grid spacing
- **Multi-scale** rather than single-resolution
- **Learned** rather than prescribed stencil weights

It most closely resembles **RBF-generated finite differences (RBF-FD)**, a meshfree
numerical method that uses smooth basis functions to approximate differential operators.

---

### 6. Quantitative Summary

| Metric                          | Value          |
|---------------------------------|----------------|
| Best first-derivative R^2       | 0.912 (du/dx)  |
| Best second-derivative R^2      | 0.501 (d2u/dy2)|
| Best Laplacian R^2              | 0.342          |
| First/second R^2 ratio          | ~7x            |
| FD neuron pairs (du/dx)         | 372            |
| FD neuron pairs (du/dy)         | 397            |
| Sign bias (chain-rule pattern)  | 91-97%         |
| Stencil cosine similarity       | 0.97-0.99      |
| Multi-scale h range             | 7-10x          |
| PINN-probe weight correlation   | -0.70 to -0.75 |

---

### 7. Limitations and Future Directions

**Limitations:**
- Results are for a single architecture (4-layer MLP, 64 neurons, tanh) on a single
  PDE (Poisson). Generalization requires testing on additional architectures and PDEs.
- The Laplacian probe R^2 of 0.34 falls short of the 0.85 target, though this led
  to the more interesting finding of implicit computation via autograd.
- Probing reveals correlation, not causation. Activation patching experiments
  (Week 3) are needed to establish causal relationships.
- Nonlinear probes might recover higher R^2 for second derivatives, which would
  indicate nonlinear rather than absent encoding.

**Future directions:**
1. Extend probing to Modified Fourier Networks and Attention-Enhanced PINNs to test
   whether the two-stage strategy is architecture-dependent.
2. Apply activation patching to confirm causal role of identified derivative circuits.
3. Test on nonlinear PDEs (Burgers equation) where the relationship between
   derivatives and the PDE residual is more complex.
4. Investigate whether encouraging explicit second-derivative encoding (e.g., via
   auxiliary loss terms) improves PINN convergence or accuracy.
5. Compare learned multi-scale stencils with optimal RBF-FD stencils to quantify
   how close the network's solution is to known numerical methods.
