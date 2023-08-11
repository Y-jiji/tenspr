mod typing;
pub use typing::*;
mod tensor;
pub use tensor::*;

#[derive(Debug)]
pub struct PreKernel {
    pub(super) graph: Vec<(Op, DType, Shape)>,
}

impl PreKernel {
    // construct a pre kernel
    pub fn new<const N: usize>(inputs: [(DType, Shape); N], f: fn ([TensorRef; N]) -> TensorRef) -> Self {
        // we have to make an internally mutuable variable for implicit recording
        let cell = std::cell::RefCell::new(PreKernel::empty());
        f(inputs.map(|(dtype, shape)| {
            let id = cell.borrow_mut().push(Op::Input, dtype, shape.clone());
            TensorRef { id, graph: &cell, shape, dtype } }));
        cell.into_inner()
    }
    // push a new tensor, used to create tensors
    pub fn push(&mut self, op: Op, dtype: DType, shape: Shape) -> usize {
        self.graph.push((op, dtype, shape));
        self.graph.len() - 1
    }
    // get the number of operators
    pub fn len(&self) -> usize { self.graph.len() }
    // make an empty graph
    fn empty() -> Self { Self { graph: vec![] } }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new_pre_kernel() {
        use DType::*;
        // we definitely need a proc-macro for this crap
        let x = PreKernel::new([(F32, sh([10, 10])), (F32, sh([10, 10]))], |[a, b]| (a * b).sum(0));
        println!("{x:?}");
    }
}