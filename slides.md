# Generating high-performance multiplatform finite element solvers from high-level descriptions

!SLIDE left

# Generating high-performance multiplatform finite element solvers from high-level descriptions

## Florian Rathgeber, Graham R. Markall, David A. Ham, Paul H. J. Kelly, Carlo Bertolli,  Adam Betts

### Imperial College London

## Mike B. Giles, Gihan R. Mudalige

### University of Oxford

## Istvan Z. Reguly

### Pazmany Peter Catholic University, Hungary

## Lawrence Mitchell

### University of Edinburgh

!SLIDE huge

# The challenge

> How do we get performance portability for the finite element method without sacrificing generality?

!SLIDE left

# The strategy

## Get the abstractions right
... to isolate numerical methods from their mapping to hardware

## Start at the top, work your way down
... as the greatest opportunities are at the highest abstraction level

## Harness the power of DSLs
... for generative, instead of transformative optimisation

!SLIDE left

# The tools

## Embedded domain-specific languages

... capture and *efficiently express characteristics* of the application/problem domain

## Active libraries

... encapsulate *specialist performance expertise* and deliver *domain-specific optimisations*

## In combination, they

* raise the level of abstraction and incorporate domain-specific knowledge
* decouple problem domains from their efficient implementation on different hardware
* capture design spaces and open optimisation spaces
* enable reuse of code generation and optimisation expertise and tool chains

!SLIDE huge

# The big picture

!SLIDE

}}} images/mapdes_abstraction_layers.png

!SLIDE huge

# Higher level abstraction

## From the equation to the finite element implementation

!SLIDE left

# Manycore Form Compiler

![The DOLFIN-FFC and Fluidity-MCFC pipelines side-by-side](images/mcfc_ffc.svg)

* Compile-time code generation, runtime coming soon
* Generates assembly and marshaling code
* Designed to handle isoparametric elements

!SLIDE left

# MCFC takes equations in UFL

## Helmholtz equation
@@@ python
f = state.scalar_fields["Tracer"]

v=TestFunction(f)
u=TrialFunction(f)

lmbda = 1
A = (dot(grad(v),grad(u))-lmbda*v*u)*dx

RHS = v*f*dx

solution = solve(A, RHS)
state.scalar_fields["Tracer"] = solution
@@@

!NOTES

## Fluidity extensions

* **`state.scalar_fields`** interfaces to Fluidity: read/write field of given name
* **`solve`** records equation to be solved and returns `Coefficient` for solution field

!SLIDE left

# ... and generates local assembly kernels

## Helmholtz OP2 kernel
@@@ c++
void A_0(double* lt, double* dt, double* c0[2], int i_r_0, int i_r_1) {
  /* Shape functions/derivatives, quadrature weights */
  double c_q0[6][2][2];
  /* Evaluate coefficients at quadrature points */
  for(int i_g = 0; i_g < 6; i_g++) {
    double ST1 = 0.0, ST0 = 0.0, ST2 = 0.0;
    ST1 += -1 * CG1[i_r_0][i_g] * CG1[i_r_1][i_g];
    double l95[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] },
                         { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    ST2 += c_q0[i_g][0][0] * c_q0[i_g][1][1]
            + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
    for(int i_d_5 = 0; i_d_5 < 2; i_d_5++) {
      for(int i_d_3 = 0; i_d_3 < 2; i_d_3++) {
        for(int i_d_9 = 0; i_d_9 < 2; i_d_9++) {
          ST0 += (l95[i_d_3][i_d_5]
            / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1]
                * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_3]
                * (l95[i_d_9][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1]
                   + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]))
               * d_CG1[i_r_1][i_g][i_d_9];
        }
      }
    }
    lt[0] += ST2 * (ST0 + ST1) * w[i_g];
  }
}
@@@

!SLIDE left float

# MCFC Pipeline

![The MCFC execution pipeline](images/mcfc_pipeline.svg)

*   ##Preprocessing

    transform integrands and spatial derivatives of forms by Jacobian determinant/inverse

*   ##Execution

    evaluate preprocessed UFL input in namespace

*   ##Form processing

    compute form metadate using UFL algorithms

*   ##Partitioning

    segment integrand according to expression depth

*   ##Expression generation

    determine integrand subexpression and insert into loop nest

!SLIDE left

# Preprocessing

## Coordinate transformations handled as part of the form
using UFL primitives `Jacobian`, `Inverse` and `Determinant`

``` python
x = state.vector_fields['Coordinate']
J = Jacobian(x)
invJ = Inverse(J)
detJ = Determinant(J)
```

* Pre-multiply each integrand by `detJ`
* Overload derivative operators:

``` python
def grad(u):
    return ufl.dot(invJ, ufl.grad(u))
```

## No special treatment of the Jacobian, its determinant or inverse

!SLIDE left

# Loop nest generation

## Loops in typical assembly kernel:

``` python
for (int i=0; i<3; ++i)       // <- basis functions
  for (int j=0; j<3; ++j)     // <- basis functions
    for (int q=0; q<6; ++q)   // <- quadrature points
      for (int d=0; d<2; ++d) // <- dimensions
```

## Inference of loop structure from preprocessed form:

* **Basis functions:** use rank of the form
* **Quadrature loop:** quadrature degree known
* **Dimension loops:** descend expression DAG, identifying maximal sub-graphs sharing a set of indices of an `IndexSum`

!SLIDE left

# Partitioning

![Partitioning the Helmholtz equation](images/helmholtz_partitioning.svg)

* Generate a subexpression for each partition
* Insert the subexpression into the loop nest depending on the indices it refers to
* Traverse the topmost expression of the form, and generate an expression that combines subexpressions, and insert into loop nest

!SLIDE huge

# Lower level abstraction

## From the finite element implementation to its efficient parallel execution

!SLIDE left

# OP2 – an active library for unstructured mesh computations

## Abstractions for unstructured grids

* **Sets** of entities (e.g. nodes, edges, faces)
* **Mappings** between sets (e.g. from edges to nodes)
* **Datasets** holding data on a set (i.e. fields in finite-element terms)

## Mesh computations as parallel loops

* execute a *kernel* for all members of one set in arbitrary order
* datasets accessed through at most one level of indirection
* *access descriptors* specify which data is passed to the kernel and how it is addressed

## Multiple hardware backends via *source-to-source translation*

* partioning/colouring for efficient scheduling and execution on different hardware
* currently supports CUDA/OpenMP + MPI - OpenCL, AVX support planned

!SLIDE left

# OP2 for finite element computations

## Finite element local assembly
... means computing the *same kernel* for *every* mesh entity (cell, facet)

## OP2 abstracts away data marshaling and parallel execution
* controls whether/how/when a matrix is assembled
* OP2 has the choice: assemble a sparse (CSR) matrix, or keep the local assembly matrices (local matrix approach, LMA)
* local assembly kernel is *translated* for and *efficiently executed* on the target architecture

## Global asssembly and linear algebra operations
... implemented as a thin wrapper on top of backend-specific linear algebra packages:  
*PETSc* on the CPU, *Cusp* on the GPU

!SLIDE left

# MCFC generates OP2 "glue code"

## OP2 host code calling the generated kernels:
@@@ c++
extern "C" void run_model_(double* dt_pointer)
{
  /* ... data marshaling */
  op_sparsity A_sparsity = op_decl_sparsity( /* ... */ );
  op_mat A_mat = op_decl_mat(A_sparsity,  /* ... */ );
  op_par_loop(A_0, "A_0", op_iteration_space(Tracer.map->from, 3, 3),
              op_arg_mat(A_mat, OP_ALL, Tracer.map, OP_ALL, Tracer.map,
                         Tracer.dat->dim, "double", OP_INC),
              op_arg_gbl(dt_pointer, 1, "double", OP_INC),
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map,
                         Coordinate.dat->dim, "double", OP_READ));
  /* ... similar for the right-hand side */
  op_solve(A_mat, RHS_vec, Tracer.dat);
  /* ... clean up */
}
@@@

!SLIDE huge

# Preliminary performance results

!SLIDE left

# Experimental setup

##Solver
CG with Jacobi preconditioning - **Dolfin:** PETSc, **OP2:** PETSc (reference), Cusp (CUDA)

##CPU
2 x 6 core Intel Xeon E5650 Westmere (HT off), 48GB RAM

##GPU
Nvidia GTX480

##Mesh
2D unstructured, 344128 triangles, square domain

##Dolfin
Revision 6739, Tensor representation, CPP optimisations on, form compiler optimisations off

!SLIDE

}}} images/mcfc_dolfin_benchmark.png

!SLIDE left

# Resources

## All the code mentioned is open source and available on *GitHub*. Try it!

## Manycore Form Compiler (MCFC)
<https://github.com/gmarkall/manycore_form_compiler>

## OP2 library
<https://github.com/OP2/OP2-Common>

## OP2 source-to-source translator
<https://github.com/OP2/OP2_ROSE_Fortran>

## This talk
<https://kynan.github.com/fenics-at-imperial-2012>
