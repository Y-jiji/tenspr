use std::{fmt::Debug, hash::Hash};

#[derive(Debug, Hash, Clone)]
pub enum RangeMath {}

#[derive(Debug, Hash, Clone)]
// reversible maps between NDRanges
pub enum NDRangeMap {}

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
    Arr(&'a Type<'a>, &'a [RangeMath]), // here usize is binder, not a length
    Tup(&'a Type<'a>, &'a Type<'a>),
    F32,
    F64,
    I32,
    I64,
}

#[derive(Debug, Hash, Clone)]
pub enum Expr<'a> {
    // tuples
    Tuple(&'a Expr<'a>, &'a Expr<'a>, Type<'a>),
    ProjL(&'a Expr<'a>, Type<'a>),
    ProjR(&'a Expr<'a>, Type<'a>),
    // indexing into array
    Index(&'a Expr<'a>, &'a Expr<'a>, Type<'a>),
    // parallel data of basic type (constructed from Array, Tuple and Primitives)
    PForGather(&'a [RangeMath], Lambda<'a, 1>, NDRangeMap, Type<'a>),
    // iterated data of basic type (constructed from Array, Tuple and Primitives)
    IForReduce(&'a [RangeMath], &'a Expr<'a>, Lambda<'a, 2>, Type<'a>),
    IForGather(&'a [RangeMath], Lambda<'a, 1>, NDRangeMap, Type<'a>),
    // let in statement
    LetIn(&'a Expr<'a>, Lambda<'a, 1>, Type<'a>),
    // binded variable
    Bind(usize, Type<'a>),
    // scalar operators
    Bin(ScalarBin, [&'a Expr<'a>;2], Type<'a>),
    Uni(ScalarUni, [&'a Expr<'a>;1], Type<'a>),
}

#[derive(Debug, Hash, Clone)]
pub struct StreamIR {
    pub vcount: usize,
}

impl StreamIR {
    pub fn sha256(&self) -> String {
        use sha2::*;
        use std::fmt::*;
        let mut name = String::new();
        write!(&mut name, "{self:?}").unwrap();
        let mut hasher = sha2::Sha256::new();
        hasher.update(name);
        let mut name = String::from("__");
        for x in hasher.finalize() {
            write!(&mut name, "{x:x}").unwrap();
        }
        return name;
    }
}
