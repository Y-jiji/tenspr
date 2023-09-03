use crate::*;

#[derive(Debug, Hash, Clone)]
pub struct Lambda<'a, const N: usize> {
    pub args: [usize; N], 
    pub expr: &'a Expr<'a>
}

#[derive(Debug, Hash, Clone)]
pub enum ScalarBin {
    Add,
    Mul,
    Sub,
    Div,
    Rem,
}

#[derive(Debug, Hash, Clone)]
pub enum ScalarUni {
    Neg,
}

#[derive(Debug, Hash, Clone)]
pub enum Type<'a> {
    Arr(&'a Type<'a>), // here usize is binder, not a length
    Tup(&'a Type<'a>, &'a Type<'a>),
    F32,
    F64,
    I32,
    I64,
}

#[derive(Debug, Hash, Clone)]
pub enum Expr<'a> {
    // tuples
    ConstUsize(usize, Type<'a>),
    Tuple(&'a [Expr<'a>], Type<'a>),
    ProjI(&'a Expr<'a>, usize, Type<'a>),
    // indexing into array
    Index(&'a Expr<'a>, &'a Expr<'a>, Type<'a>),
    // uninitialized array of given size
    Array(&'a [Expr<'a>], Type<'a>),
    // parallel compute
    PForGather(&'a [Expr<'a>], &'a [Expr<'a>], Lambda<'a, 2>, &'a Expr<'a>, Type<'a>),
    // iterative compute
    IForGather(&'a [Expr<'a>], &'a [Expr<'a>], Lambda<'a, 2>, &'a Expr<'a>, Type<'a>),
    IForReduce(&'a [Expr<'a>], Lambda<'a, 2>, &'a Expr<'a>, Type<'a>),
    // let in statement
    LetIn(&'a Expr<'a>, Lambda<'a, 1>, Type<'a>),
    // binded variable
    Bind(usize, Type<'a>),
    // scalar operators
    Bin(ScalarBin, [&'a Expr<'a>; 2], Type<'a>),
    Uni(ScalarUni, [&'a Expr<'a>; 1], Type<'a>),
}

impl<'a> Expr<'a> {
    pub fn get_type(&self) -> Type<'a> {
        use Expr::*;
        let (Tuple(.., t) | ProjI(.., t) | Index(.., t) | Array(.., t) | PForGather(.., t) | IForGather(.., t) | IForReduce(.., t) | LetIn(.., t) | Bind(.., t) | Bin(.., t) | Uni(.., t) | ConstUsize(.., t)) = self;
        return t.clone();
    }
}