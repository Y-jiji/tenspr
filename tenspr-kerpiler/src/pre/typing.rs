#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DType {
    F32, 
    F64,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Dim {Var(usize), Fix(usize)}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Shape(Vec<Dim>);

impl Shape {
    pub fn remove(mut self, dim: usize) -> Self {
        self.0.remove(dim);
        return self;
    }
}

pub fn sh<const N: usize>(x: [usize; N]) -> Shape {
    Shape(x.map(|i| Dim::Fix(i)).into())
}