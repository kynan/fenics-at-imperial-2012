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

!SLIDE

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

* raise the level ob abstraction and incorporate domain-specific knowledge
* decouple problem domains from their efficient implementation on different hardware
* capture design spaces and open optimisation spaces
* enable reuse of code generation and optimisation expertise and tool chains

!SLIDE left

# Active libraries open optimisation spaces

![placeholder](http://placehold.it/1024x600&text=Applications, active libraries, hardware platforms)

!SLIDE

# Higher level abstraction

## From the equation to the finite element implementation

!SLIDE left

# Manycore Form Compiler

![placeholder](http://placehold.it/800x360&text=MCFC/Fluidity vs. DOLFIN/FFC)

* Compile-time code generation, runtime coming soon
* Generates assembly and marshaling code
* Designed to handle isoparametric elements

!SLIDE left float

# MCFC Pipeline

![placeholder](http://placehold.it/400x640&text=MCFC Pipeline)

* **Preprocessing:** transform integrands and spatial derivatives of forms by Jacobian determinant/inverse
* **Execution:** evaluate preprocessed UFL input in namespace
* **Form processing:** compute form metadate using UFL algorithms
* **Partitioning:** segment integrand according to expression depth
* **Expression generation:** determine integrand subexpression and insert into loop nest

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

![placeholder](http://placehold.it/800x400&text=Partitioning example: Helmholtz)

* Generate a subexpression for each partition
* Insert the subexpression into the loop nest depending on the indices it refers to
* Traverse the topmost expression of the form, and generate an expression that combines subexpressions, and insert into loop nest

!SLIDE

# Lower level abstraction

## From the finite element implementation to its efficient parallel execution

!SLIDE left

# OP2 – an active library for unstructured mesh computations

## Abstractions for unstructured grids

* **Sets** of entities (e.g. nodes, edges, faces)
* **Mappings** between sets (e.g. from edges to nodes)
* **Datasets** holding data on a set (i.e. a field in finite element terms)

## Mesh computations as parallel loops

* execute a *kernel* for all members of one set in arbitrary order
* datasets accessed through at most one level of indirection
* access descriptors specify which data is passed to the kernel and how it is addressed
* partioning and colouring for efficient scheduling and execution on different hardware architectures (CUDA/OpenMP + MPI)

!SLIDE left

# Why OP2 for finite element computations?

## Finite element local assembly
... means computing the *same kernel* for *every* mesh entity (cell, facet)

## OP2 abstracts away the matrix representation
* matrix is a dataset with a pair of associated maps for rows and columns of the matrix resp.
* controls whether/how/when the matrix is assembled
* takes care of efficient execution of  the local assembly kernel

## Global asssembly and linear algebra operations
... implemented as a thin wrapper on top of backend-specific linear algebra packages:
PETSc on the CPU, Cusp on the GPU

OP2 has the choice: assemble a sparse (CSR) matrix or keep the local assembly matrices (local matrix approach, LMA)

!SLIDE

# Chaining the tools together

## From the equation to performance portable code for different hardware architectures

!SLIDE left

# MCFC takes equations in a UFL "dialect"

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

## UFL extensions

* **`state.scalar_fields`** interfaces to Fluidity: read/write field of given name
* **`solve`** records equation to be solved and returns `Coefficient` for solution field

!NOTES

@@@ python
t = state.scalar_fields["Tracer"]
v = TestFunction(t)
u = TrialFunction(t)
a = v*u*dx
L = v*t*dx
new_t = solve(a, L)
state.scalar_fields["Tracer"] = new_t
@@@

!SLIDE left

# ... and generates local assembly kernels

## Helmholtz OP2 kernel
@@@ c++
void A_0(double* localTensor, double* dt, double* c0[2], int i_r_0, int i_r_1) {
  /* Shape functions/derivatives, quadrature weights */
  double c_q0[6][2][2];
  /* Evaluate coefficient at quadratur points */
  for(int i_g = 0; i_g < 6; i_g++) {
    double ST1 = 0.0;
    double ST0 = 0.0;
    double ST2 = 0.0;
    ST1 += -1 * CG1[i_r_0][i_g] * CG1[i_r_1][i_g];
    double l95[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    double l35[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    ST2 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
    for(int i_d_5 = 0; i_d_5 < 2; i_d_5++) {
      for(int i_d_3 = 0; i_d_3 < 2; i_d_3++) {
        for(int i_d_9 = 0; i_d_9 < 2; i_d_9++) {
          ST0 += (l35[i_d_3][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_3] * (l95[i_d_9][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_9];
        }
      }
    }
    localTensor[0] += ST2 * (ST0 + ST1) * w[i_g];
  }
}
@@@

!NOTES

@@@ c++
void mass(double* localTensor, double* c0[2], int i_r_0, int i_r_1)
{
  const double CG1[3][6] = { ... };
  const double d_CG1[3][6][2] = { ... };
  const double w[6] = { ... };
  double c_q0[6][2][2];
  for(int i_g = 0; i_g < 6; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        };
      };
    };
  };
  for(int i_g = 0; i_g < 6; i_g++)
  {
    double ST0 = 0.0;
    ST0 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g]
           * (c_q0[i_g][0][0] * c_q0[i_g][1][1]
               + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0] += ST0 * w[i_g];
  };
}
@@@

!SLIDE left

# ... and OP2 "glue code" to run the model

@@@ c++
extern "C" void run_model_(double* dt_pointer)
{
  void* state = get_state();
  op_field_struct Coordinate = extract_op_vector_field(state, "Coordinate", 10, 0);
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 6, 0);
  op_sparsity A_sparsity = op_decl_sparsity(Tracer.map, Tracer.map, "A_sparsity");
  op_mat A_mat = op_decl_mat(A_sparsity, Tracer.dat->dim, "double", 8, "A_mat");
  op_par_loop(A_0, "A_0", op_iteration_space(Tracer.map->from, 3, 3),
              op_arg_mat(A_mat, OP_ALL, Tracer.map, OP_ALL, Tracer.map,
                         Tracer.dat->dim, "double", OP_INC),
              op_arg_gbl(dt_pointer, 1, "double", OP_INC),
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map,
                         Coordinate.dat->dim, "double", OP_READ));
  op_dat RHS_vec = op_decl_vec(Tracer.dat, "RHS_vec");
  op_par_loop(RHS_0, "RHS_0", Tracer.map->from,
              op_arg_dat(RHS_vec, OP_ALL, Tracer.map, Tracer.dat->dim,
                         "double", OP_INC),
              op_arg_gbl(dt_pointer, 1, "double", OP_INC),
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map,
                         Coordinate.dat->dim, "double", OP_READ),
              op_arg_dat(Tracer.dat, OP_ALL, Tracer.map, Tracer.dat->dim,
                         "double", OP_READ));
  op_solve(A_mat, RHS_vec, Tracer.dat);
  op_free_vec(RHS_vec);
  op_free_mat(A_mat);
}
@@@

!SLIDE

# Conclusion

## The MAPDES vision

!SLIDE left

}}} images/mapdes_abstraction_layers.png
