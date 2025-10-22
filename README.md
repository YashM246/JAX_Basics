# JAX Basics

A personal learning repository for exploring the JAX framework - Google's library for high-performance numerical computing and automatic differentiation.

## About

This repository contains Jupyter notebooks with code examples, experiments, and detailed explanations as I learn JAX. The notebooks progress from basic concepts to more advanced topics, including common gotchas and state management patterns.

## Notebooks

### 1. [JAX Warmup](1_JAX_warmup.ipynb)
Introduction to JAX fundamentals:
- NumPy-like syntax and array operations
- Immutable arrays and the `.at()` syntax
- Random number generation with explicit key management
- AI accelerator agnostic computing (GPU/TPU)
- Core transform functions:
  - `jit()` - Just-in-time compilation for performance
  - `grad()` - Automatic differentiation (including Jacobians and Hessians)
  - `vmap()` - Automatic vectorization across batches

### 2. [JAX Basics](2_JAX_basics.ipynb)
Deep dive into JAX fundamentals and common pitfalls:
- Understanding pure functions and why they matter
- **Gotcha #1**: Pure Functions - side effects and global state
- **Gotcha #2**: In-Place Updates - using `.at()` for array modifications
- **Gotcha #3**: Out-of-Bounds Indexing - non-error behavior
- **Gotcha #4**: Non-array Inputs - strict type requirements
- **Gotcha #5**: Random Numbers - explicit PRNG key management
- **Gotcha #6**: Control Flow - JIT compilation considerations

### 3. [JAX Basics II](3_JAX_basics_II.ipynb)
Advanced patterns and state management:
- The problem of state in functional programming
- Why stateful code breaks with JIT compilation
- Implementing stateful patterns (e.g., counters) in JAX
- **PyTrees** - JAX's way of handling complex data structures
  - Understanding PyTree structure and leaves
  - Manipulating PyTrees with `tree.map`
  - Handling gradients for models with many parameters
- Foundation for building neural networks in JAX

### 4. [JAX MLP](4_JAX_MLP.ipynb)
Building and training a Multi-Layer Perceptron from scratch:
- Initializing neural network parameters with `init_mlp_params`
- **He Initialization** - proper weight initialization for deep networks
  - Understanding exploding and vanishing activations
  - Why `sqrt(2/n_in)` scaling matters
  - Keeping variance stable across layers
- **Forward propagation** with ReLU activation
- **Loss function** - Mean Squared Error (MSE)
- **Gradient descent** - automatic differentiation with `jax.grad()`
- **Training loop** - iterative parameter updates
- **PyTree magic** - updating all parameters with `tree.map()`
- Practical examples: learning y = x² and y = sin(3x)

![MLP Training Result](jax_mlp_pred_sin.png)
*Successfully trained MLP learning a sine function*

### 5. [JAX Custom PyTrees](5_JAX_Custom_PyTrees.ipynb)
Understanding and creating custom PyTree nodes:
- Why JAX can't traverse custom classes by default
- **Custom PyTree registration** with `register_pytree_node()`
- Implementing `flatten` and `unflatten` functions
  - Children (trainable parameters) vs auxiliary data (metadata)
  - How JAX decomposes and reconstructs objects
- **Common gotcha**: Tuples as PyTree containers vs leaves
  - Understanding tree structure of shape tuples
  - Solutions: `is_leaf` parameter and direct mapping
- Foundation for building custom neural network layers
- Essential for creating reusable ML components

### 6. [JAX Parallelism](6_JAX_Parallelism.ipynb)
Parallel computing patterns in JAX:
- Coming soon...

## Topics Covered

- ✅ Basic JAX operations and NumPy compatibility
- ✅ Automatic differentiation with `grad`, `jacfwd`, `jacrev`
- ✅ Just-in-time compilation with `jit`
- ✅ Vectorization with `vmap`
- ✅ Random number generation with PRNG keys
- ✅ Pure functional programming patterns
- ✅ State management in functional style
- ✅ PyTrees for complex data structures
- ✅ Common gotchas and how to avoid them
- ✅ Neural network parameter initialization (He initialization)
- ✅ MLP forward propagation with ReLU activation
- ✅ Loss functions (Mean Squared Error)
- ✅ Gradient descent optimization with `tree.map()`
- ✅ Complete training loops
- ✅ Custom PyTree registration for custom classes
- ✅ Flatten/unflatten functions for PyTree nodes
- ✅ PyTree gotchas (tuples as containers vs leaves)
- 🔄 Parallel computing with `pmap` and `vmap` (in progress)
- 🔄 Advanced architectures and optimizers (in progress)

## Setup

```bash
# Basic installation
pip install jax jaxlib

# For visualization examples
pip install matplotlib
```

For GPU support:
```bash
pip install --upgrade "jax[cuda12]"
```

## Key Takeaways

- JAX arrays are **immutable** - use `.at()` methods instead of in-place operations
- JAX requires **pure functions** for transformations like `jit` and `grad`
- Random numbers require **explicit key management** - no global state
- State must be **explicitly passed** as function arguments and return values
- JIT compilation provides significant **performance improvements**
- Same code runs on **CPU, GPU, or TPU** without modification

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/google/jax)
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [JAX Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
