use crate::pre::*;
use smallvec::*;

// memory layout configuration (provided by a backend)
// a layout is memory units and wih connections between them
pub struct Layout {
    // size of each node available (or register)
    size: Vec<usize>,
    // links to other layout nodes
    edge: Vec<Vec<(usize, usize)>>,
}

// predicates or data formats on different dimensions
// + data format is how a tensor is stored in memory
// + predicates are about how tensor is computed
pub enum Predicate {
    Sparse(usize),      // sparse on one dimension (something like csr or csc format)
    Dense,              // fully dense on all dimensions
}

// kernel builder that optimize a graph
pub struct KernelBuilder {
    // a pre-kernel to work on
    prekernel: PreKernel,
    // additional predicates on each tensor
    predicate: Vec<SmallVec<[Predicate; 1]>>,
    // forward links for quick lookup
    fwd_links: Vec<SmallVec<[usize; 1]>>,
}

// how to allocate memory resources to a tensor? (with temporal streaming in mind)
// [?] memory segment as registers and movement is multi-layered, memory pyramid
impl KernelBuilder {
    // make a new kernel builder from a recorded prekernel
    fn new(prekernel: PreKernel) -> KernelBuilder {
        let n = prekernel.len();
        let mut fwd_links = (0..n).map(|_| SmallVec::new()).collect::<Vec<_>>();
        for (i, (op, _, _)) in prekernel.graph.iter().enumerate() {
            for j in op.upstream() { fwd_links[j].push(i); }
        }
        KernelBuilder {
            prekernel, fwd_links,
            predicate: (0..n).map(|_| SmallVec::new()).collect(), 
        }
    }
    // thread model and joining patterns
    // tiled iteration
}