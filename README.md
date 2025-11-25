# Noncontextual Assignment Polytopes for GPT Scenarios

This repository implements a small, self-contained pipeline for testing **noncontextuality** in generalized probabilistic theory (GPT) scenarios using **assignment polytopes** and a linear-program feasibility test.

The code is designed to be:

- mathematically transparent,
- minimal (almost no defensive checks),
- highly readable thanks to explicit variable names and heavy use of NumPy / SciPy / SymPy / cdd.

---

## 1. What this repo does

Given a finite GPT experiment specified by:

- a set of **states** `states`,
- a set of **effects** `effects`,
- a set of **transformations** `transformations`,

the pipeline does the following:

1. **Lumps transformations** into states and effects to build equivalent PM scenarios.
2. **Computes data tables**  
   - PTM: \(p(k \mid s, t)\)  
   - PM from lumped states  
   - PM from lumped effects
3. **Finds operational identities** (linear relations) among states, effects, and transformations via exact nullspaces.
4. **Builds assignment polytopes**:
   - source-assignment polytope (preparations),
   - effect-assignment polytope (measurements).
5. **Builds a linear system** \(M x = b\) encoding the F1 formulation of a noncontextual ontological model.
6. **Tests feasibility** of \(M x = b\) with \(x \ge 0\) using linear programming.
   - If feasible: the data admit a noncontextual model of this type.
   - If infeasible: the scenario is contextual (in that sense).

All of this is orchestrated from `orchestrator.py`.

---

## 2. Installation

### 2.1. Dependencies

Python 3.10+ is recommended.

Core Python packages:

- `numpy`
- `scipy` (for `scipy.optimize.linprog`)
- `sympy`
- `pycddlib` (or your system’s `cdd` Python bindings)

Install with e.g.:

```bash
pip install numpy scipy sympy pycddlib
