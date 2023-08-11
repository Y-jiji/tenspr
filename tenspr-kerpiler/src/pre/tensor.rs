use super::*;
use std::ops::*;
use smallvec::*;

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Input,
    Add{lhs: usize, rhs: usize},
    Mul{lhs: usize, rhs: usize},
    Sum{oprand: usize, dim: usize},
    Prd{oprand: usize, dim: usize},
}

impl Op {
    pub fn upstream(&self) -> SmallVec<[usize; 2]> {
        match *self {
            Op::Input => smallvec![],
            Op::Add{lhs, rhs} => smallvec![lhs, rhs],
            Op::Mul{lhs, rhs} => smallvec![lhs, rhs],
            Op::Sum{oprand, ..} => smallvec![oprand],
            Op::Prd{oprand, ..} => smallvec![oprand],
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

// implment reduce operator for tensor reference
macro_rules! impl_red_op {
    ($Name: ident, $name: ident) => {
        impl<'a> TensorRef<'a> {
            pub fn $name(&self, dim: usize) -> TensorRef<'a> {
                let mut pre = self.graph.borrow_mut();
                let shape = self.shape.clone().remove(dim);
                let id = pre.push(Op::$Name { oprand: self.id, dim }, self.dtype, shape.clone());
                drop(pre);
                return TensorRef { id, graph: self.graph, shape, dtype: self.dtype }
            }
        }
    };
}

impl_red_op!(Sum, sum);
impl_red_op!(Prd, prd);

// implment binary operator for tensor reference
macro_rules! impl_bin_op {
    ($Name: ident, $name: ident) => {
        impl<'a> $Name for TensorRef<'a> {
            type Output = TensorRef<'a>;
            fn $name(self, rhs: Self) -> Self::Output {
                let mut pre = self.graph.borrow_mut();
                assert!(self.dtype == rhs.dtype, "todo");
                assert!(self.shape == rhs.shape, "todo");
                let id = pre.push(Op::$Name { lhs: self.id, rhs: rhs.id }, self.dtype, self.shape.clone());
                drop(pre);
                return TensorRef { id, graph: self.graph, shape: self.shape.clone(), dtype: self.dtype }
            }
        }
        impl<'a> $Name for &TensorRef<'a> {
            type Output = TensorRef<'a>;
            fn $name(self, rhs: Self) -> Self::Output {
                let mut pre = self.graph.borrow_mut();
                assert!(self.dtype == rhs.dtype, "todo");
                assert!(self.shape == rhs.shape, "todo");
                let id = pre.push(Op::$Name { lhs: self.id, rhs: rhs.id }, self.dtype, self.shape.clone());
                drop(pre);
                return TensorRef { id, graph: self.graph, shape: self.shape.clone(), dtype: self.dtype }
            }
        }
    };
}

impl_bin_op!(Add, add);
impl_bin_op!(Mul, mul);