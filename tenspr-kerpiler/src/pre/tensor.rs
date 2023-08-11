use super::*;
use smallvec::*;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
// binary operation that satisfies assication law
pub enum _Op_ {
    Add,
    Mul,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
// unary operation
pub enum Op_ {
    FNeg,
    FInv,
    INeg,
    BNeg,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Op {
    Input,
    Bin { opr: _Op_, lhs: usize, rhs: usize },
    Red { opr: _Op_, opd: usize, dim: usize },
    Map { opr: Op_, opd: usize },
    Mov { opd: usize, map: SmallVec<[usize; 2]> },
}

impl Op {
    pub fn upstream(&self) -> SmallVec<[usize; 1]> {
        match self {
            Op::Input => smallvec![],
            Op::Bin { lhs, rhs, .. } => smallvec![*lhs, *rhs],
            Op::Map { opd, .. } |
            Op::Red { opd, .. } |
            Op::Mov { opd, .. } => smallvec![*opd],
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorRef<'a> {
    pub(super) graph: &'a std::cell::RefCell<PreKernel>,
    pub(super) shape: Shape,
    pub(super) dtype: DType,
    pub(super) id: usize,
}