---
layout: page
title: "Linear Algebra"
date: 2020-08-14 15:00:00 -0000
categories: Math
---
- [Basic Concepts](#basic-concepts)
  - [Vectors and Matrix](#vectors-and-matrix)
  - [Vector Space](#vector-space)
  - [Orthogonality](#orthogonality)
  - [Determinant](#determinant)
  - [Eigen Value & Eigen Vector](#eigen-value--eigen-vector)
  - [Diagonalization](#diagonalization)
  - [Singular Value Decomposition](#singular-value-decomposition)
  - [Linear Transformation](#linear-transformation)
  - [Complex Matrices](#complex-matrices)
- [Linear Algebra for Data Sciece](#linear-algebra-for-data-sciece)
  - [Matrix Calculus](#matrix-calculus)
  
## Basic Concepts

### Vectors and Matrix

- vectors
- linear system and matrix
  - geometric - linear combination
  - operation/transformation
- matrix operations
- invertible and singular matrix
  - [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix)
- Gaussian elimination
  - Echelon matrix
  - pivot variables, \(\# = C\(A\)\), free variable\(\# = N\(A\)\)
  - [elementary matrix](https://en.wikipedia.org/wiki/Elementary_matrix)
- Null matrix
- matrix factorization
  - [LU Decomposition](https://en.wikipedia.org/wiki/LU_decomposition)

### Vector Space

- vector space - definition
  - commutative, associative, distributive, unique identity
  - uniqueness of identity and inverse
- subspace
  - row space C\(A^T\)
    - pivot rows
  - column space C\(A\)
    - **Ax = b*- &lt;=&gt; b in C\(A\)
  - null space N\(A\)/kernel
    - **Ax = 0**
  - Solution of Linear System 
    - pivot variables, free variables
    - Null space matrix
    - special solution 
    - left null space N\(A^T\)
  - m x n matrix by rank r
    - r = m = n : square and invertible
    - r = m &lt; n : short and wide: infinity number of solutions
    - r = n &lt; m : Tall and thin, 0 or 1 solution
    - r &lt; m, n : Not full rank, 0 or infinity solutions
  - rank
    - number of pivot column = dim C\(A\)
    - freedom = \# of free columns = dim N\(A\)
- span
  - Linearly independent
- basis
  - **basis of matrix spaces and function spaces**
- dimension

### Orthogonality

- **[Fundamental Theorem of Linear Algebra](https://en.wikipedia.org/wiki/Fundamental_theorem_of_linear_algebra)**
  - orthogonal subspaces
    - row space and null space
    - column space and left null space
- projection
  - projection to vector - [projection matrix](https://mathworld.wolfram.com/ProjectionMatrix.html) \($$\frac{\mathbf{aa^T}}{\mathbf{a^Ta}}$$\)
  - projection to subspace
    - orthogonal to C\(A\) - A^T\(b-Ax\) = 0
    - [hat matrix/projection matrix ](https://en.wikipedia.org/wiki/Projection_matrix): A\(A^TA\)^-1A^T
- [Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
  - orthonormal
  - orthogonal matrix Q
- **[QR Decomposition](https://en.wikipedia.org/wiki/Gram%E2%80%93QR_Decomposition)**
- Matrix Space, function space
  - rank 1 matrix

### Determinant

- basic properties
  - I -&gt; 1
  - exchange rows -&gt; reverse sign
  - linear in row 1\(i\) by itself
  - two rows equal, its determinant is 0
  - t times row i add to row j does not change
  - row of all zeros -&gt; 0
  - triangular matrix determinant -&gt; product of diagonal
  - A is **singular*- then det A = 0
  - det BA = \(det B\) \- det\(A\)
  - det \(A^T\) = det A
- calculation
  - big formula
  - cofactors formula
- Cramer's rule 
- Cross product, outter product/Volume

### Eigen Value & Eigen Vector

- transform invariability
- properties
  - linearity - add
  - multiply - P^k, lambda ^ k
  - sum = trace, product \(=det\(A\)\) \(= product of pivots\)
  - det\(A-\lambda I\)
  - complex eigen value
  - commuting matrices share eigen values
- Eigen Space
  - geometric multiplicity
  - algebraic multiplicity
- Differential equations
  - linear differential equation system
- Markov matrices
- Fourier Series

### Diagonalization

- Diagonalizable matrix
  - eigen space dimension = number of independent eigen vectors \(geometirc multiplicity\)
  - repetitions of eigen values = n roots of $$det(A - \lambda I ) = 0$$ \(characteristic polynomial\)
- Real matrix
  - eigen value and vector comes in conjugate pairs
- Real Symmetric matrix
  - [Spectral Theorem](https://en.wikipedia.org/wiki/Spectral_theorem) 
    - real eigen values \(all diagonailzable\)
    - eigen vectors can be chosen orthonormal \($$Q\Lambda Q^T$$\)
  - [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)
- Positive definite matrices
  - x^T A x &gt; 0
    - positive eigen values 
    - pivots positive
    - principal submatrices positive definite
    - det\(A\) = 0 \(invertible\)
  - Cholesky R^T R with independent columns
- Positive semidefinite Matrices
  - **0 eigen value, depedent columns**
  - unit square root -&gt; Cholesky
- [Similarity](https://en.wikipedia.org/wiki/Matrix_similarity)
  - B = M^-1 A M
  - **same eigen value, trace, determinant, rank, eigen space dimenstion, **
  - different Null, column space, **eigen vectors**, sigular values
  - [**Jordan Form**](https://en.wikipedia.org/wiki/Jordan_normal_form), Jordan Block
    - Generalized Eigenvector

### Singular Value Decomposition

- $$A = U \Sigma V^T = \sum \mathbf{u_i} \sigma_i \mathbf{v_i^T}$$
  - $$AV = U \Sigma$$
- basis
  - V: first r from row space \(eigen values of A^TA\), n - r from null space
  - U: r from column space\(eigenvalues of AA^T\), n - r from left null space
- singular vectors
- in case of positive semidefinite\(orthonormal eigen vector, non negative eigen value\), same as eigen decomposition

### Linear Transformation

- range: column space
- kernel: null space
- linearity based on basis - represent by matrix
  - eg. projection, integral, derivative, rotation
  - Wavelet transform
  - Discrete Fourier Transform
    - Fourier Matrix
    - Fast Fourier

### Complex Matrices

- Hermitian
- Inner product in complex space

## Linear Algebra for Data Sciece

### [Matrix Calculus](https://en.wikipedia.org/wiki/Matrix_calculus)

- $\frac{\partial y}{\partial x}$, $\frac{\partial \mathbf{y}}{\partial x}$, $\frac{\partial \mathbf{Y}}{\partial x}$ just list normal derivatives by column
  
- $$\mathbf{a} = \mathbf{XW}, \frac{\partial \mathbf{a}}{\partial W} = \mathbf{X}^T$$
  
- $\frac{\partial y}{\mathbf{x}},\frac{\partial y}{\mathbf{X}}$ list the result according to denominator
  
- Jacobian $(x_1, ... , x_n) \rightarrow (h_1,...h_m)$

$$\frac{\partial \mathbf{h}}{\partial \mathbf{x}} =
  \begin{pmatrix}
				\frac{\partial h_1 }{\partial x} & \cdots & \frac{\partial h_1 }{\partial x_n}\\ 
				\vdots & \ddots & \vdots \\
				\frac{\partial{h_m}}{\partial x_1 } & \cdots & \frac{\partial h_m }{\partial x_n}
	- \end{pmatrix}$$

- For matrix to matrix, vector to matrix, more than one dimension tensors as result (result must include all results and can be used with Chain Rule)

Formulas

$$ \frac{\partial \mathbf{a^T x}}{\mathbf{x}} = \frac{\partial  \mathbf{xa}}{\mathbf{x}} = \mathbf{a}$$
$$ \frac{\partial \mathbf{A x}}{\mathbf{x}} = \mathbf{A}$$
$$\frac{\partial \mathbf{ x^T A x}}{\partial{\mathbf{x}}}=( \mathbf{A} +  \mathbf{A^T} )\mathbf{x}$$ 
$$\frac{\partial \mathbf{ x^T A x}}{\partial{\mathbf{x}} \partial{\mathbf{x}}}
	= 2\mathbf{A} $$
$$\frac{\partial( \mathbf{AX} +  b) \mathbf{C} ( \mathbf{DX} + e)}{\partial{\mathbf{x}}} = \mathbf{A^TC}(\mathbf{DX} + e) + \mathbf{D^TC^T} ( \mathbf{AX} +  b) $$