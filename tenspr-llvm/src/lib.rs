use inkwell::{
    basic_block::BasicBlock,
    values::{AnyValueEnum, IntValue, AnyValue, StructValue, VectorValue, BasicValueEnum, BasicValue, FunctionValue},
    types::{BasicTypeEnum, BasicMetadataTypeEnum}, AddressSpace,
};
use std::{collections::HashMap, cell::RefCell};
use kerpiler::{ScalarBin, Type, Expr};

// road map:
// 1. build jit functions without parallelism
// 2. support vectorized computation
// 3. link external libraries for multi-threading support

const X86: u8 = 0;

pub struct LLVMBackend<'llvm, const _t: u8> {
    llvm_context: &'llvm inkwell::context::Context,
    llvm_module: inkwell::module::Module<'llvm>,
    llvm_engine: inkwell::execution_engine::ExecutionEngine<'llvm>,
}

pub struct LLVMFunction();

impl<'ctx> LLVMBackend<'ctx, X86> {
    pub fn new(name: &str, llvm_context: &'ctx inkwell::context::Context) -> Self {
        use inkwell::OptimizationLevel::None;
        let llvm_module = llvm_context.create_module(name);
        let llvm_engine = llvm_module.create_jit_execution_engine(None).unwrap();
        LLVMBackend {
            llvm_context,
            llvm_module,
            llvm_engine,
        }
    }
    pub fn to_llvm_type(&self, t: Type) -> BasicTypeEnum<'ctx> {
        match t {
            Type::Arr(x) => match self.to_llvm_type(x.clone()) {
                BasicTypeEnum::ArrayType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::FloatType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::IntType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::PointerType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::StructType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::VectorType(x) => x.ptr_type(AddressSpace::default()).into(),
            },
            Type::Tup(x, y) => {
                let x = self.to_llvm_type(x.clone());
                let y = self.to_llvm_type(y.clone());
                self.llvm_context.struct_type(&[x, y], false).into()
            }
            Type::F32 => self.llvm_context.f32_type().into(),
            Type::F64 => self.llvm_context.f64_type().into(),
            Type::I32 => self.llvm_context.i32_type().into(),
            Type::I64 => self.llvm_context.i64_type().into(),
        }
    }
    pub fn compile<'a>(&self, inputs_binds: Vec<usize>, inputs_types: Vec<Type<'a>>, expr: Expr<'a>) -> FunctionValue<'ctx> {
        use inkwell::types::BasicType;
        let return_ty = self.to_llvm_type(expr.get_type());
        let inputs_ty = inputs_types.into_iter().map(|x| self.to_llvm_type(x).into()).collect::<Vec<BasicMetadataTypeEnum>>();
        let builder = LLVMFunctionBuilder::new(self.llvm_context, self.llvm_context.create_builder(), 
            self.llvm_module.add_function("function", return_ty.fn_type(inputs_ty.as_slice(), false), None), inputs_binds);
        builder.finalize(expr)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Holder<'llvm> {
    value: BasicValueEnum<'llvm>,
    block: BasicBlock<'llvm>,
}

pub struct LLVMFunctionBuilder<'llvm> {
    llvm_context: &'llvm inkwell::context::Context,
    llvm_builder: inkwell::builder::Builder<'llvm>,
    function: inkwell::values::FunctionValue<'llvm>,
    count: std::cell::RefCell<usize>,
    binds: HashMap<usize, Holder<'llvm>>,
    entry: BasicBlock<'llvm>,
}

impl<'llvm> LLVMFunctionBuilder<'llvm> {
    pub fn new(llvm_context: &'llvm inkwell::context::Context, llvm_builder: inkwell::builder::Builder<'llvm>, function: inkwell::values::FunctionValue<'llvm>, inputs_binds: Vec<usize>) -> Self {
        let entry = llvm_context.append_basic_block(function, "entry");
        let mut this = Self {
            llvm_builder, llvm_context, function, count: RefCell::new(0), binds: HashMap::new(), entry,
        };
        for (i, x) in inputs_binds.into_iter().enumerate() {
            let param = this.function.get_nth_param(i as u32).unwrap();
            this.binds.insert(x, Holder { value: param, block: entry });
        }
        return this;
    }
    pub fn name(&self) -> String {
        *self.count.borrow_mut() += 1;
        format!("_{}", *self.count.borrow())
    }
    pub fn to_llvm_type(&self, t: Type) -> BasicTypeEnum<'llvm> {
        match t {
            Type::Arr(x) => match self.to_llvm_type(x.clone()) {
                BasicTypeEnum::ArrayType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::FloatType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::IntType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::PointerType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::StructType(x) => x.ptr_type(AddressSpace::default()).into(),
                BasicTypeEnum::VectorType(x) => x.ptr_type(AddressSpace::default()).into(),
            },
            Type::Tup(x, y) => {
                let x = self.to_llvm_type(x.clone());
                let y = self.to_llvm_type(y.clone());
                self.llvm_context.struct_type(&[x, y], false).into()
            }
            Type::F32 => self.llvm_context.f32_type().into(),
            Type::F64 => self.llvm_context.f64_type().into(),
            Type::I32 => self.llvm_context.i32_type().into(),
            Type::I64 => self.llvm_context.i64_type().into(),
        }
    }
    pub fn bb(&mut self) -> BasicBlock<'llvm> {
        let n = self.name();
        self.llvm_context.append_basic_block(self.function, &n)
    }
    pub fn finalize(mut self, expr: kerpiler::Expr) -> inkwell::values::FunctionValue<'llvm> {
        let holder = self.build(expr, self.entry);
        self.llvm_builder.position_at_end(holder.block);
        self.llvm_builder.build_return(Some(&holder.value));
        self.function
    }
    pub fn build(&mut self, expr: kerpiler::Expr, mut from: BasicBlock<'llvm>) -> Holder<'llvm> {
        // validation: 
        // + apart from binders, if a value is available at the `from` block, it should be also avaiable in evaluated expression
        // + all active binders (binded values in binds table) should be available in from block
        // + for every two binded values, one of them should be available in the block correspondent to another
        use kerpiler::Expr::*;
        match expr {
            Bind(i, _t) => {
                let Holder { value, block } = self.binds[&i];
                let bid = |bb: BasicBlock<'llvm>| -> usize {
                    bb.get_name().to_str().unwrap()[1..].parse::<usize>().unwrap()
                };
                let block = if bid(block) < bid(from) { from } else { block };
                Holder { value, block }
            },
            LetIn(inp, lam, _t) => {
                let x = self.build(inp.clone(), from);
                self.binds.insert(lam.args[0], x);
                let holder = self.build(lam.expr.clone(), x.block);
                self.binds.remove(&lam.args[0]); holder
            },
            ConstUsize(size, _t) => {
                Holder { value: self.llvm_context.i64_type().const_int(size as u64, false).as_basic_value_enum(), block: from }
            }
            Tuple(xs, _t) => {
                let mut block = from;
                let mut txs = vec![];
                let mut vxs = vec![];
                for x in xs {
                    let y = self.build(x.clone(), block);
                    block = y.block;
                    txs.push(y.value.get_type());
                    vxs.push(y.value);
                }
                self.llvm_builder.position_at_end(block);
                let tz = self.llvm_context.struct_type(&txs, false);
                let pz = self.llvm_builder.build_alloca(tz, &self.name());
                let vz = self.llvm_builder.build_load(pz, &self.name()).into_struct_value();
                for (i, vx) in vxs.into_iter().enumerate() {
                    self.llvm_builder.build_insert_value(vz, vx, i as u32, &self.name());
                }
                Holder { value: vz.into(), block }
            },
            ProjI(x, i, _t) => {
                let x = self.build(x.clone(), from);
                let vx = x.value.into_struct_value();
                self.llvm_builder.position_at_end(x.block);
                let value = self.llvm_builder.build_extract_value(vx, i as u32, &self.name()).unwrap();
                Holder { value, block: x.block }
            },
            Index(arr, idx, _t) => {
                let arr = self.build(arr.clone(), from);
                let idx = self.build(idx.clone(), arr.block);
                self.llvm_builder.position_at_end(arr.block);
                let value = unsafe { self.llvm_builder.build_gep(arr.value.into_pointer_value(), &[idx.value.into_int_value()], &self.name()) };
                Holder { value: value.into(), block: idx.block }
            },
            Array(dims, t) => {
                let i64t = self.llvm_context.i64_type();
                let size = dims.iter().fold(i64t.const_int(1, false), |a, x| {
                    let x = self.build(x.clone(), from);
                    from = x.block;
                    return self.llvm_builder.build_int_mul(a, x.value.into_int_value(), &self.name())
                });
                self.llvm_builder.position_at_end(from);
                let Type::Arr(inner_t) = t else { panic!("fuck! not an array type") };
                let value = self.llvm_builder.build_array_malloc(self.to_llvm_type(inner_t.clone()), size, &self.name()).unwrap();
                Holder { value: value.into(), block: from }
            },
            IForGather(dims, offs, lam, ptr, _t) => {
                let i64t = self.llvm_context.i64_type();
                let dims = dims.iter().map(|x| {
                    let x = self.build(x.clone(), from);
                    from = x.block;
                    return x.value;
                }).collect::<Vec<_>>();
                let offs = offs.iter().map(|x| {
                    let x = self.build(x.clone(), from);
                    from = x.block;
                    return x.value;
                }).collect::<Vec<_>>();
                // build loops of multi-layers
                let exit_block = self.bb();
                let mut head_block = self.bb();
                let mut body_block = self.bb();
                self.llvm_builder.position_at_end(from);
                self.llvm_builder.build_unconditional_branch(head_block);
                let mut gidx = i64t.const_zero();
                let mut idxs = vec![];
                for i in 0..dims.len() {
                    self.llvm_builder.position_at_end(body_block);
                    self.llvm_builder.build_unconditional_branch(head_block);
                    let exit_block = if i != 0 { head_block } else { exit_block };
                    head_block = if i != 0 { body_block } else { head_block };
                    body_block = if i != 0 { self.bb()  } else { body_block };
                    self.llvm_builder.position_at_end(head_block);
                    let iter = self.llvm_builder.build_phi(i64t, &self.name());
                    iter.add_incoming(&[
                        (&i64t.const_zero(), from), 
                        (&iter.as_basic_value().into_int_value().const_add(i64t.const_int(1, false)), body_block)
                    ]);
                    let iter = iter.as_basic_value().into_int_value();
                    idxs.push(iter);
                    let cond = self.llvm_builder.build_int_compare(
                        inkwell::IntPredicate::NE, iter, dims[i].into_int_value(), &self.name());
                    self.llvm_builder.build_conditional_branch(cond, body_block, exit_block);
                    gidx = self.llvm_builder.build_int_nuw_add(
                        gidx, self.llvm_builder.build_int_nuw_mul(iter, offs[i].into_int_value(), &self.name()), &self.name());
                }
                self.llvm_builder.position_at_end(body_block);
                // pass aggregated value as args[0]
                let aggt = self.llvm_context.struct_type(&vec![i64t.into(); idxs.len()], false);
                let aggv = self.llvm_builder.build_load(self.llvm_builder.build_alloca(aggt, &self.name()), &self.name());
                for (i, idx) in idxs.into_iter().enumerate() {
                    self.llvm_builder.build_insert_value(aggv.into_struct_value(), idx, i as u32, &self.name());
                }
                self.binds.insert(lam.args[0], Holder { value: aggv, block: body_block });
                // pass pointer array as args[1]
                let ptr = self.build(ptr.clone(), from);
                self.llvm_builder.position_at_end(ptr.block);
                let ptr_value = unsafe { self.llvm_builder.build_gep(ptr.value.as_basic_value_enum().into_pointer_value(), &[gidx], &self.name()) };
                self.binds.insert(lam.args[1], Holder { value: ptr_value.into(), block: ptr.block });
                // head block
                let body = self.build(lam.expr.clone(), body_block);
                self.llvm_builder.position_at_end(body.block);
                self.llvm_builder.build_unconditional_branch(head_block);
                // pass out an evalutated value
                Holder { value: ptr.value, block: exit_block }
            }
            Uni(operator, expr, _t) => {
                todo!()
            }
            Bin(operator, expr, t) => {
                let a = self.build(expr[0].clone(), from);
                let b = self.build(expr[1].clone(), a.block);
                self.llvm_builder.position_at_end(b.block);
                let v = match (operator, t) {
                    (ScalarBin::Add, Type::F32 | Type::F64) => self.llvm_builder.build_float_add(a.value.into_float_value(), b.value.into_float_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Sub, Type::F32 | Type::F64) => self.llvm_builder.build_float_sub(a.value.into_float_value(), b.value.into_float_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Mul, Type::F32 | Type::F64) => self.llvm_builder.build_float_mul(a.value.into_float_value(), b.value.into_float_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Div, Type::F32 | Type::F64) => self.llvm_builder.build_float_div(a.value.into_float_value(), b.value.into_float_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Add, Type::I32 | Type::I64) => self.llvm_builder.build_int_add(a.value.into_int_value(), b.value.into_int_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Sub, Type::I32 | Type::I64) => self.llvm_builder.build_int_sub(a.value.into_int_value(), b.value.into_int_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Mul, Type::I32 | Type::I64) => self.llvm_builder.build_int_mul(a.value.into_int_value(), b.value.into_int_value(), &self.name()).as_basic_value_enum(),
                    (ScalarBin::Div, Type::I32 | Type::I64) => self.llvm_builder.build_int_signed_div(a.value.into_int_value(), b.value.into_int_value(), &self.name()).as_basic_value_enum(),
                    _ => unimplemented!(),
                };
                Holder { value: v, block: b.block }
            }
            _ => todo!()
        }
    }
}