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
- Best practices for managing state in neural networks

## Topics Covered

- ✅ Basic JAX operations and NumPy compatibility
- ✅ Automatic differentiation with `grad`, `jacfwd`, `jacrev`
- ✅ Just-in-time compilation with `jit`
- ✅ Vectorization with `vmap`
- ✅ Random number generation with PRNG keys
- ✅ Pure functional programming patterns
- ✅ State management in functional style
- ✅ Common gotchas and how to avoid them
- 🔄 Neural network implementations (in progress)
- 🔄 GPU/TPU acceleration patterns (in progress)

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
