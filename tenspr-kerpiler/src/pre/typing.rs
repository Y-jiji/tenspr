use smallvec::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DType {
    F32, 
    F64,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Dim {Var(usize), Fix(usize)}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Shape(SmallVec<[Dim; 2]>);

impl Shape {
    pub fn remove(mut self, dim: usize) -> Self {
        self.0.remove(dim); self
    }
    pub fn insert(mut self, dim: usize, n: usize) -> Self {
        self.0.insert(dim, Dim::Fix(n)); self
    }
}

pub fn sh<const N: usize>(x: [usize; N]) -> Shape {
    Shape(x.map(|i| Dim::Fix(i)).to_smallvec())
}