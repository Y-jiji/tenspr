# Tenspr

Another WIP on sparse tensor computation. 

# Core Abstractions

## Hardware backend

### Storage Format

A storage format represented in offset arithmetics, load, store and allocate operations in particular address spaces. 

### Worker Group

A worker group provides address spaces and workers that can access these address space. 

### Worker

A worker is an ALU and registers attached to it. 

## GraphIR

### Operators

Operators are classified into binary, unary, nullary and reduce. 

### Tensor Expr

Tensor is treated as a scalar expression containing indices as free variables. 

### Index Ordering

Index ordering decides whether a computation can be lazy, and on each level it should be materalized. 

Index also decides whether an iteration should be parelleized. 

### Worker Group

A worker group is attached to each operator. 

### Storage Format

For each materialization level, a storge format is assigned to produce an materialization plan.

## Optimization

### Storage Format Search

Storage format can be symbolically searched in this framework using MemIR. 

### Index Splitting

An index can be splitted to support tiling. 

### Common Expression

Common expressions can be either pre-computed or re-computed. 
