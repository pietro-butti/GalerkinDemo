# Stein–Poisson / Galerkin for correlators: basic formulas

Reference note for A and B. Not read by the test-writer (C): every spec copies
what it needs, so that it is self-contained.

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $z\in\mathbb R^{V}$ | field configuration, $V=T N_s$, sites $(t,x)$, periodic |
| $p(z)=e^{-S(z)}/Z$ | base distribution |
| $\langle h\rangle=\int p\,h$, $\delta h=h-\langle h\rangle$ | mean and centred part |
| $\mathrm{Cov}(f,g)=\langle\delta f\,\delta g\rangle$ | inner product, $\lVert f\rVert^2=\mathrm{Var}f$ |
| $F=\nabla S$ | force |
| $\epsilon$ | **source strength** (called $\lambda$ in the note) |
| $\lambda$ | reserved for the $\phi^4$ quartic coupling |
| $\Pi(t)=\sum_x z(t,x)$ | slab sum, $t=0,\dots,T-1$ (Julia index $t+1$) |

Renaming the source strength $\lambda\to\epsilon$ avoids the clash with the quartic
coupling $\lambda(z^2-1)^2$ in the interacting spec.

## 1. Correlator by reweighting

### 1.1 Base and target action

$$S(z)\quad\text{(base)},\qquad S_\epsilon(z)=S(z)-\epsilon\,O_0(z)\quad\text{(target)},\qquad p_\epsilon=e^{-S_\epsilon}/Z_\epsilon .$$

The source operator is $O_0=\Pi(0)$ (zero-momentum field on timeslice 0).

### 1.2 Correlator as a response, and the deformation

$$\partial_\epsilon\langle\Pi(t)\rangle_\epsilon\Big|_{\epsilon=0}=\mathrm{Cov}\big(\Pi(t),O_0\big)=C(t).$$

The first-order change of the density is

$$p_\epsilon=p\,\big[1+\epsilon\,\Delta+O(\epsilon^2)\big],\qquad \boxed{\Delta=\delta O_0=O_0-\langle O_0\rangle}\quad(\text{source at }0).$$

In the $\mathbb Z_2$-symmetric phase $\langle\Pi\rangle=0$, so $\Delta=\Pi(0)$.
The standard estimator is $U_t=\Delta\,\Pi(t)$, with $\langle U_t\rangle=C(t)$.

### 1.3 Transport ansatz

Map $z\mapsto z'=z+\epsilon f(z)$ with $f=\nabla\Phi$. Its Jacobian is $1+\epsilon\nabla\!\cdot f$, so the pushed-forward density is

$$p'(z)=p(z)\big[1-\epsilon\,(\nabla\!\cdot f-f\cdot\nabla S)\big]+O(\epsilon^2).$$

Matching $p'=p_\epsilon$ at first order gives $-(\nabla\!\cdot f-f\cdot\nabla S)=\Delta$.

### 1.4 Stein operators

$$\mathcal T g=\nabla\!\cdot g-g\cdot\nabla S\quad(\text{vector field }g),\qquad
\mathcal L\Phi=\mathcal T\nabla\Phi=\nabla^2\Phi-F\cdot\nabla\Phi\quad(\text{scalar}).$$

Properties (integration by parts, $\nabla\!\cdot(p\,g)=p\,\mathcal T g$):

$$\langle\mathcal T g\rangle=0,\qquad \langle\mathcal L h\rangle=0,\qquad
\mathrm{Cov}(f,\mathcal L g)=-\langle\nabla f\cdot\nabla g\rangle,\qquad
\mathcal T(h\,g)=h\,\mathcal T g+g\cdot\nabla h .$$

### 1.5 Stein–Poisson equation

$$\boxed{\mathcal L\Phi+\Delta=0}\qquad\Longleftrightarrow\qquad \mathcal T f+\Delta=0,\ \ f=\nabla\Phi .$$

**Sign check.** With $S_\epsilon=S-\epsilon O_0$ one has $p_\epsilon\simeq p(1+\epsilon\Delta)$, and the
transport gives $p(1-\epsilon\,\mathcal T f)$. Hence $\mathcal T f=-\Delta$: the sign is $+\Delta$ in
$\mathcal L\Phi+\Delta=0$, consistent with the note.

### 1.6 Residual and improved estimator

$$\boxed{R=\Delta+\mathcal L\Phi},\qquad f_t:=\nabla\Phi\cdot\nabla\Pi(t)=\sum_x\frac{\partial\Phi}{\partial z(t,x)} .$$

$$\boxed{\hat A_t=U_t+\mathcal T\big(\Pi(t)\nabla\Phi\big)=R\,\Pi(t)+f_t}\qquad(\text{Leibniz rule}).$$

- $\langle\hat A_t\rangle=C(t)$ **for every $\Phi$**, because $\langle\mathcal T g\rangle=0$. The choice of $\Phi$ never biases the result.
- If $R\equiv0$ (exact solution $\Phi^\star$), then $\hat A_t=f_t=C(t)$ on every configuration: zero variance.
- In general $\mathrm{Var}\,\hat A_t\neq\mathrm{Var}\,R$. If $R$ and $\Pi(t)$ decorrelate at large $t$ and $f_t$ fluctuates little, $\mathrm{Var}\,\hat A_t\approx\langle R^2\rangle\langle\Pi(t)^2\rangle$.
- For the scalar estimator $\hat O=O+\mathcal L\Phi$ one has exactly $\hat O-\langle O\rangle=R$, so $\mathrm{Var}\,\hat O=\mathrm{Var}\,R$.

## 2. Galerkin decomposition

### 2.1 Basis expansion

Choose $M$ functions $\psi_i$ and write

$$\Phi=\sum_{j=1}^M\alpha_j\psi_j,\qquad
\Delta=\sum_{j=1}^M s_j\,\delta\psi_j,\qquad \delta\psi_j=\psi_j-\langle\psi_j\rangle .$$

(The sum for $\Delta$ is exact only if $\Delta$ lies in the span; otherwise one uses the projections $d,c$ below.)
The residual is linear in $\alpha$:

$$R(\alpha)=\sum_j s_j\,\delta\psi_j+\sum_j\alpha_j\,\mathcal L\psi_j .$$

### 2.2 Conditions

$R\equiv0$ is a continuum of conditions, so we impose $M$ of them with test functions $\chi_i$:

$$\mathrm{Cov}(\chi_i,R)=0,\qquad
\chi_i=\psi_i\ \ (\text{Galerkin}),\qquad \chi_i=\mathcal L\psi_i\ \ (\text{minimum variance}).$$

### 2.3 Matrices

$$B_{ij}=\mathrm{Cov}(\psi_i,\psi_j),\qquad
A^{\rm grad}_{ij}=\langle\nabla\psi_i\cdot\nabla\psi_j\rangle,$$
$$A^{\rm ibp}_{ij}=-\mathrm{Cov}(\psi_i,\mathcal L\psi_j),\qquad
Q_{ij}=\mathrm{Cov}(\mathcal L\psi_i,\mathcal L\psi_j).$$

$A^{\rm grad}$ is a plain average, **not** a covariance. $A^{\rm grad}$ and $A^{\rm ibp}$ have the same expectation $A$ (symmetric, positive semidefinite), but differ on a finite sample. $B$, $Q$ are symmetric.

### 2.4 Solvers

Galerkin ($\chi_i=\psi_i$): $\mathrm{Cov}(\psi_i,R)=(Bs)_i-(A\alpha)_i=0$, so

$$\boxed{\alpha_{\rm grad}=(A^{\rm grad})^{-1}Bs,\qquad \alpha_{\rm ibp}=(A^{\rm ibp})^{-1}Bs}$$

Minimum variance ($\chi_i=\mathcal L\psi_i$): $\mathrm{Cov}(\mathcal L\psi_i,R)=-\big((A^{\rm ibp})^\top s\big)_i+(Q\alpha)_i=0$, so

$$\boxed{\alpha_{\rm vm}=Q^{-1}\big(A^{\rm ibp}\big)^\top s}$$

Interpretation: $\alpha_{\rm vm}$ minimises $\mathrm{Var}\,R(\alpha)=\mathrm{Var}\,\Delta+2\alpha^\top c+\alpha^\top Q\alpha$ with $c_i=\mathrm{Cov}(\mathcal L\psi_i,\Delta)$, i.e. least squares of $-\Delta$ on the features $\mathcal L\psi_j$. The Galerkin solution minimises the Dirichlet energy $\tfrac12\alpha^\top A\alpha-\alpha^\top Bs$.

### 2.5 Residual: closed span and leakage

If $\mathcal L\psi_j=-\sum_i\Lambda_{ij}\,\delta\psi_i$ pointwise (closed span), then

$$R=\sum_i(s-\Lambda\alpha)_i\,\delta\psi_i,\qquad
\hat A^{\rm ibp}=\hat B\Lambda,\quad \hat Q=\Lambda^\top\hat B\Lambda,\quad
\alpha_{\rm ibp}=\alpha_{\rm vm}=\Lambda^{-1}s,\quad R\equiv0 .$$

These identities hold on **any sample**, because they are pointwise. In general, $\mathcal L\psi_j=-\sum_i\Lambda_{ij}\delta\psi_i+\eta_j$ with leakage $\eta_j\perp\mathrm{span}$, $\Lambda=B^{-1}A$, and

$$\mathrm{Var}\,R=(s-\Lambda\alpha)^\top B\,(s-\Lambda\alpha)+\alpha^\top\Gamma\,\alpha,\qquad \Gamma=Q-A^\top B^{-1}A\succeq0 .$$

Galerkin zeroes the first term; minimum variance minimises the sum, so $\mathrm{Var}\,R_{\rm vm}\le\mathrm{Var}\,R_{\rm gal}$.

### 2.6 Sample version

Replace $\langle\cdot\rangle$ and $\mathrm{Cov}$ by sample averages and covariances with $1/(N-1)$ normalisation. Fit $\alpha$ on an offline subset and apply it on an independent online subset (cross-fitting). Fitting and applying on the same configurations introduces an $O(1/N)$ bias. Solve with `\`, never `inv`.

## 3. Free theory

### 3.1 Action, force, generator

$$S=\tfrac12 z^\top Kz,\qquad K=2I-2\kappa H,\qquad F=Kz,\qquad \langle zz^\top\rangle=K^{-1},$$

where $H$ is the nearest-neighbour hopping matrix of the periodic $T\times N_s$ lattice (the $\lambda=0$ limit of the $\phi^4$ action; requires $\kappa<1/4$).
$$\mathcal L=\nabla^2-Kz\cdot\nabla .$$

### 3.2 Slab reduction

Let $v_j$ be the indicator of slab $j$, so $\Pi(j)=v_j\cdot z$. A slab-constant vector has two spatial neighbours at every site, hence

$$Kv_j=\sum_iK^{(0)}_{ij}v_i,\qquad
K^{(0)}_{ij}=(2-4\kappa)\,\delta_{ij}-2\kappa\,(\delta_{i,j+1}+\delta_{i,j-1})\ \ (\text{periodic in }t),$$

$$G^{(0)}=(K^{(0)})^{-1},\qquad K_k=2-4\kappa-4\kappa\cos\!\frac{2\pi k}{T}\ \ (\text{eigenvalues of }K^{(0)}),\qquad
C(t)=N_s\,G^{(0)}_{t0}.$$

### 3.3 Level-1 basis ($M=T$)

$$\psi_j=\Pi(j),\quad \frac{\partial\psi_j}{\partial z(t,x)}=\delta_{tj},\quad \nabla\psi_i\cdot\nabla\psi_j=N_s\,\delta_{ij},\quad \nabla^2\psi_j=0,$$

$$\mathcal L\psi_j=-\sum_xF(j,x)=-\sum_iK^{(0)}_{ij}\,\psi_i\qquad(\Lambda=K^{(0)},\ \text{closed span}).$$

Since $\langle\psi_j\rangle=0$: $\delta\psi_j=\psi_j$, $\Delta=\Pi(0)=\psi_0$, so $\boxed{s=e_0}$.

### 3.4 Population matrices

$$B=N_s\,G^{(0)},\qquad A^{\rm grad}=A^{\rm ibp}=N_s\,I,\qquad Q=N_s\,K^{(0)} .$$

### 3.5 Exact solution

$$\alpha^\star=\Lambda^{-1}s=G^{(0)}_{\cdot0},\qquad
\Phi^\star=\sum_uG^{(0)}_{u0}\,\Pi(u),\qquad f^\star(t,x)=G^{(0)}_{t0},\qquad
f^\star_t=N_sG^{(0)}_{t0}=C(t),\qquad R\equiv0 .$$

For a linear basis, $f_t=\sum_j\alpha_j\sum_x\partial_{z(t,x)}\psi_j=N_s\,\alpha_t$ for any $\alpha$, so

$$\hat A_t=R\,\Pi(t)+N_s\,\alpha_t .$$

### 3.6 Sample identities (any sample with $N>T$)

$$\hat A^{\rm grad}=N_s I\ \ (\text{exactly, no fluctuation}),\qquad
\hat A^{\rm ibp}=\hat BK^{(0)},\qquad \hat Q=K^{(0)}\hat BK^{(0)},$$

$$\hat\alpha_{\rm ibp}=\hat\alpha_{\rm vm}=G^{(0)}_{\cdot0}\ \ (\text{exact on every sample}),\qquad
\hat\alpha_{\rm grad}=\frac{\hat B_{\cdot0}}{N_s}\ \ (\text{inherits the noise of }\hat B).$$

### 3.7 Residuals

$$R_{\rm ibp}=R_{\rm vm}\equiv0\ \Rightarrow\ \hat A_t=N_sG^{(0)}_{t0}=C(t)\ \text{on every configuration}.$$

For grad, $\delta\alpha=\hat\alpha_{\rm grad}-\alpha^\star=\delta b/N_s$ with $\delta b=(\hat B-B)_{\cdot0}$, and

$$R_{\rm grad}=-\sum_i\big(K^{(0)}\delta\alpha\big)_i\,\Pi(i),\qquad
\mathrm{Var}\,R_{\rm grad}\simeq\frac1{N_s}\,\delta b^\top K^{(0)}\delta b .$$

For Gaussian samples, $\mathbb E[\delta b\,\delta b^\top]=(B\,B_{00}+B_{\cdot0}B_{\cdot0}^\top)/N$, which gives

$$\mathbb E\,\frac{\mathrm{Var}\,R_{\rm grad}}{\mathrm{Var}\,\Delta}\simeq\frac{T+1}{N}\quad(\alpha\text{ fitted on an independent sample of size }N).$$

This last number is a derived prediction, to be calibrated numerically before it enters a spec as an acceptance interval.

### 3.8 Summary for the three $\alpha$

| | equation | free theory, one sample | $\mathrm{Var}\,R/\mathrm{Var}\,\Delta$ |
|---|---|---|---|
| grad | $A^{\rm grad}\alpha=Bs$ | $\hat B_{\cdot0}/N_s$ (noisy) | $\simeq(T+1)/N$ |
| ibp | $A^{\rm ibp}\alpha=Bs$ | $G^{(0)}_{\cdot0}$ (exact) | $0$ |
| vm | $Q\alpha=(A^{\rm ibp})^\top s$ | $G^{(0)}_{\cdot0}$ (exact) | $0$ |