//! # Attributes: 
//! 
//! When properly augmented with attributes, 
//! an expression (abstracting away all the free variables, btw.) will represent a tensor. 
//! 
//! Which free variable is abtracted away first is provided by Attr.index. 
//! 
//! Also, we will need a lower level language for storage and access (storage operation will compile to it. )
//! For each tensor, we seperate storage and computation with the following abstraction. 
//! 
//! Hopefully, if we change attributes of one tensor, 
//! the amortized cost on only a constant number of tensors will be affected. 
//! 
//! ## Index ordering:
//! 
//! Index ordering determines the order of computation. 
//! For example, if an expression `A` has two free variables, say, `i` and `j`. 
//! Ordering `i, j` means A is evaluated as `FOR i: FOR j: A` . 
//! 
//! ## Storage:
//! 
//! First off, index ordering for one expression is unique and thus 
//! for any computation that access some operand, some intermediate structure 
//! should be constructed and stored to allow different index ordering requested by downstream operators. 
//! 
//! It is quite obvious that larger intermidiate structures always causes slower computation. 
//! 
//! Therefore, the storage format is provided in a fallback pattern: 
//! if an index ordering mismatches an ordering requested by downstream operators, 
//! its storage dimension must increase to satisfy the reordering. 
//! 
//! Since our ultimate goal is to create cost estimation directly for physical machines, 
//! storage format should include physical device details. 
//! 
//! ## Computation: 
//! 
//! Computation is distributed over workers. 
//! A worker is a set of register and an ALU, with a bus connected to some ram or hard disk. 
//! 
//! A worker group is a set of multi-indexed workers associated with a storage plan. 
//! 
//! A sequence of worker groups are assigned to computation. 
//! 
//! In one time step, one worker can be only used once, which means work groups with interections can only start one after another. 
//! 
//! ## sequential number: 
//! 
//! Each operator is assigned a unique sequential number. When two operators are rushing on a single worker group. 
//! The operator with bigger sequential number will first be computed. 
//! 
//! # Optimizations
//! 
//! ## Index splitting: 
//! 
//! To support tiling optimization, we should enable splitting one index into two. 
//! 
//! However, this will bring use problems that some splitting might have some remains. 
//! 
//! This means some index is typed by another index. 
//! 
//! When a index is typed by another, it cannot be precedent before another. 
//! 
//! ## Algebraic optimization
//! 
//! Seperation of common coefficients, etc. 

use smallvec::{SmallVec, smallvec};
use std::fmt::Debug;

/// Index reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IRef(usize);

/// Index arithmetics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Idx {
    Const(usize),
    Variable(usize),
    Div(IRef, IRef),
    Rem(IRef, IRef),
    Mul(IRef, IRef),
}

pub fn imerge<const N: usize>(lhs: &[IRef], rhs: &[IRef]) -> SmallVec<[IRef; N]> 
    where [IRef; N] : smallvec::Array<Item=IRef>,
{
    let mut rhs = rhs.iter().copied().peekable();
    let mut lhs = lhs.iter().copied().peekable();
    let mut x = SmallVec::new();
    while lhs.len() != 0 && rhs.len() != 0 {
        let l = lhs.peek().unwrap();
        let r = rhs.peek().unwrap();
        x.push(
            if l < r { lhs.next().unwrap() }
            else if l > r { rhs.next().unwrap() }
            else { lhs.next().unwrap(); rhs.next().unwrap() }
        );
    }
    x.extend(lhs.chain(rhs));
    return x;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XRef(usize);

/// Expressions that are conceptually scalar. 
/// However, if we abstract away free variables (who are always idxs), we readily find it is a tensor. 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expr {
    Bin{op: BinOp, lhs: XRef, rhs: XRef},
    Uni{op: UniOp, rhs: XRef},
    Gen{op: NilOp},
    Red{op: RedOp, idx: usize, init: XRef, expr: XRef},
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // basic arithmetic
    Add, Mul, Sub, Div, Rem, 
    // tuple and conditional projection
    Tup, PjC,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniOp {
    // left and right projection of a tuple
    PjL, PjR,
    // bitwise negation
    Neg,
    // 
    Cast(Type),
}

/// Nullary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NilOp {
    // input
    Input(usize),
    // const
    Const(usize),
    // index
    Index(IRef),
}

/// Reduce operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedOp {
    Add, Mul, Max, Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TRef(usize);

/// a type marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    F32, F64,
    I32, I64,
    Bool,
    Sized(XRef),
    Tuple(TRef, TRef)
}

macro_rules! arith {
    () => {Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Sized(_)};
}

macro_rules! integer {
    () => {Type::I32 | Type::I64 | Type::Sized(_)};
}

macro_rules! primitive {
    () => {Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Sized(_) | Type::Bool};
}

#[derive(Debug, Clone)]
pub struct Attr<'a> {
    pub seqnr: usize,
    pub index: SmallVec<[IRef; 5]>,
    pub store: SmallVec<[&'a dyn StorageFormat; 5]>,
    pub group: &'a dyn WorkerGroup,
}

pub struct MRef(usize);
pub struct Addr(usize);

pub enum MemIR {
    // integer (usually) calculation
    Calc(XRef),
    // load something from an array with given offset
    Load(Addr, MRef),
    // append the result of an expression
    Append(Addr, MRef),
    // allocate an extendable array with given size
    Alloca(XRef),
    // abort an operation or make the cost estimation explode
    Revert,
}

pub trait StorageFormat where Self: Debug {
    // Initializing a tensor with given shape. 
    // Input shape and type, and Returns allocation requests
    fn init(&self, shape: &[XRef], ty: Type) -> (&[XRef], Type);
    // Write an element at given position. 
    fn rand_write(&self, index: &[XRef], value: XRef) -> MRef;
    // Write an element at given position. 
    fn rand_write_cost(&self, index: &[XRef], value: XRef) -> MRef;
    // Create a write cursor at given position
    fn iter_write_init(&self, index: &[XRef]) -> MRef;
    // This will write a value to cursor position and move the cursor to next position. 
    // Cursor should be returned value from iter_read_init or this method. 
    fn iter_write_next(&self, cursor: MRef, index: &[XRef], value: XRef) -> MRef;
    // Return write cost. 
    fn iter_write_cost(&self) -> usize;
    // Perform a random access. 
    fn rand_read(&self, index: &[XRef]) -> MRef;
    // Returns random read cost. 
    fn rand_read_cost(&self, index: &[XRef]) -> MRef;
    // Initialize an iterator with an index. 
    // Let's call it `I`. Suppose the storged tensor is A. 
    // This function will create an iterator over A[I:], giving all coordinates. 
    fn iter_read_init(&self, index: &[XRef]) -> MRef;
    // Compute next element from an iterator. 
    // Cursor should be returned value from iter_read_init or this method. 
    // Returns (coordinate, cursor, value)
    fn iter_read_next(&self, cursor: MRef) -> (&[MRef], MRef, MRef);
    // Return read cost
    fn iter_read_cost(&self) -> usize;
}

// WorkerGroup is directly connected to a physical backend. 
// Therefore it is also in charge of lowering memory r/w commands. 
pub trait WorkerGroup where Self: Debug {
    // Allocate a dense tensor of given length
    // Returns a handle of the address
    fn allocate(&self, op: XRef, ty: Type) -> Addr;
    // Computational capacity that have some hierarchical structure. 
    // Each level can join partially. 
    fn capacity(&self) -> usize;
}

// Backend is an object capable of giving workergroups and other stuff
pub trait Backend where Self: Debug {
    // default worker group
    fn wg_default(&self) -> &dyn WorkerGroup;
    // default storage format
    fn sf_default(&self) -> &dyn StorageFormat;
}

pub struct Ctxt<'a> {
    pub inputs: Vec<(TRef, Attr<'a>)>,          // input list
    pub idx_graph: Vec<Idx>,                // graph of indexes
    pub ops_graph: Vec<(TRef, Expr, Attr<'a>)>, // graph of operators
    pub typ_graph: Vec<Type>,                   // type references
    pub ret_value: Vec<XRef>,
    pub fwd_links: Vec<Vec<XRef>>,              // downstream links
    pub buf_memir: Vec<MemIR>,                  // buffer for memory ir
    pub backend: &'a dyn Backend,
}

impl<'a> Ctxt<'a> {
    // dereference an expression
    pub fn xderef(&self, xref: XRef) -> &(TRef, Expr, Attr<'a>) {
        let XRef(xref) = xref; &self.ops_graph[xref]
    }
    // dereference a type
    pub fn tderef(&self, tref: TRef) -> &Type {
        let TRef(tref) = tref; &self.typ_graph[tref]
    }
    // forward links for expression
    pub fn xfwd_links_mut(&mut self, xref: XRef) -> &mut Vec<XRef> {
        let XRef(xref) = xref; &mut self.fwd_links[xref]
    }
    // forward links for expression
    pub fn xfwd_links(&self, xref: XRef) -> &Vec<XRef> {
        let XRef(xref) = xref; &self.fwd_links[xref]
    }
    // emit an expression as output
    pub fn emit(&mut self, x: XRef) {
        self.ret_value.push(x);
    }
    // estimate amortized time cost for an expression
    pub fn tcost(&self, x: XRef) -> usize {
        todo!()
    }
    // estimate amortized memory cost for an expression
    pub fn mcost(&mut self, x: XRef) -> IRef {
        let fwd_links = self.xfwd_links(x);
        let index_x = &self.xderef(x).2.index;
        let mut len_prefix = index_x.len();
        for link in fwd_links {
            let index_y = &self.xderef(*link).2.index;
            for i in 0..len_prefix {
                if index_x[i] != index_y[i] { len_prefix = i; break; }
            }
        }
        let suffix = index_x[len_prefix..].to_owned();
        suffix[1..].into_iter().copied().fold(suffix[0], |a, b| {
            self.idx_graph.push(Idx::Mul(a, b));
            IRef(self.idx_graph.len() - 1)
        })
    }
    // put an index and get an index reference
    pub fn index(&mut self) -> IRef {
        // index reference
        todo!()
    }
    // add input with index ordering
    pub fn input(&mut self, typ: Type, idx: &[IRef]) -> XRef {
        // reference
        let tref = {self.typ_graph.push(typ); TRef(self.typ_graph.len() - 1)};
        // make default attribute
        let attr = Attr {
            index: idx.into(),
            seqnr: self.ops_graph.len(),
            group: self.backend.wg_default(),
            store: smallvec![self.backend.sf_default(); idx.len()],
        };
        // reference to input argument
        let aref = self.inputs.len();
        // push an operator
        self.ops_graph.push((tref, Expr::Gen { op: NilOp::Input(aref) }, attr));
        // return expression reference
        return XRef(self.ops_graph.len() - 1);
    }
    // add a new unary operator
    pub fn map(&mut self, op: UniOp, rhs: XRef) -> Result<XRef, Expr> {
        // deref rhs expression
        let (tref, _, attr) = self.xderef(rhs);
        // create the expected expression
        let x = Expr::Uni { op, rhs };
        // keep the same index ordering
        let i = attr.index.clone();
        // create
        match op {
            UniOp::Cast(x) => {
            },
            UniOp::Neg => {
            },
            _ => {}
        }
        todo!("build type and expression")
    }
    // add a new binary operator
    pub fn bin(&mut self, op: BinOp, lhs: XRef, rhs: XRef) -> Result<XRef, Expr> {
        // deref lhs and rhs expressions
        let (tref_l, _, attr_l) = self.xderef(lhs);
        let (tref_r, _, attr_r) = self.xderef(rhs);
        // deref lhs and rhs types
        let (type_l, type_r) = (self.tderef(*tref_l).clone(), self.tderef(*tref_r).clone());
        // create the expected expression
        let x = Expr::Bin { op, lhs, rhs };
        // merge distinct indices from upstream, keep them sorted
        let i = imerge(&attr_l.index, &attr_r.index);
        // default attributes before any optimization
        let a = Attr {
            // using default worker group
            group: self.backend.wg_default(),
            // using default storage format
            store: smallvec![self.backend.sf_default(); i.len()],
            // default sequential number
            seqnr: self.ops_graph.len(), index: i, 
        };
        // do type check for operator & infer output type from operator
        let t = match op {
            // arithmetic operators
            BinOp::Add | BinOp::Div | BinOp::Mul | BinOp::Sub 
                if !matches!(type_l, arith!()) || type_l != type_r => Err(x)?,
            BinOp::Rem if !matches!(type_l, integer!()) || type_l != type_r => Err(x)?,
            BinOp::Add | BinOp::Mul | BinOp::Sub | BinOp::Div | BinOp::Rem => type_l,
            // make a tuple
            BinOp::Tup => Type::Tuple(*tref_l, *tref_r),
            // conditional projection
            BinOp::PjC => {
                // left hand side should be bool
                if type_l != Type::Bool { Err(x)? };
                // right hand side should be a tuple of the same type
                let Type::Tuple(type_rl, typr_rr) = type_r else { Err(x)? };
                if self.tderef(type_rl) != self.tderef(typr_rr) { Err(x)? };
                *self.tderef(type_rl)
            }
        };
        // push the type and get a reference
        let t = {self.typ_graph.push(t); TRef(self.typ_graph.len() - 1)};
        // push operator with built types
        self.ops_graph.push((t, x, a));
        let x = XRef(self.ops_graph.len() - 1);
        // make operands point to this
        self.xfwd_links_mut(lhs).push(x);
        self.xfwd_links_mut(rhs).push(x);
        // return reference to expression
        Ok(x)
    }
}