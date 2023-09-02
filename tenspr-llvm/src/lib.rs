use inkwell::{
    basic_block::BasicBlock,
    values::{AnyValueEnum, IntValue, AnyValue, StructValue, VectorValue, BasicValueEnum},
    types::BasicTypeEnum,
};
use std::collections::HashMap;

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
}

impl<'llvm> LLVMFunctionBuilder<'llvm> {
    pub fn name(&self) -> String {
        *self.count.borrow_mut() += 1;
        format!("_{}", *self.count.borrow())
    }
    pub fn bb(&mut self) -> BasicBlock<'llvm> {
        let n = self.name();
        self.llvm_context.append_basic_block(self.function, &n)
    }
    pub fn build(&mut self, expr: kerpiler::Expr, from: BasicBlock<'llvm>) -> Holder<'llvm> {
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
            Tuple(x, y, _t) => {
                let x = self.build(x.clone(), from);
                let y = self.build(y.clone(), x.block);
                let tz = self.llvm_context.struct_type(&[x.value.get_type(), y.value.get_type()], false);
                let pz = self.llvm_builder.build_alloca(tz, &self.name());
                let vz = self.llvm_builder.build_load(pz, &self.name()).into_struct_value();
                self.llvm_builder.build_insert_value(vz, x.value, 0, &self.name());
                self.llvm_builder.build_insert_value(vz, y.value, 1, &self.name());
                Holder { value: vz.into(), block: y.block }
            },
            ProjL(x, _t) => {
                let x = self.build(x.clone(), from);
                let vx = x.value.into_struct_value();
                let value = self.llvm_builder.build_extract_value(vx, 0, &self.name()).unwrap();
                Holder { value, block: x.block }
            },
            ProjR(x, _t) => {
                let x = self.build(x.clone(), from);
                let vx = x.value.into_struct_value();
                let value = self.llvm_builder.build_extract_value(vx, 1, &self.name()).unwrap();
                Holder { value, block: x.block }
            },
            Index(arr, idx, _t) => {
                let arr = self.build(arr.clone(), from);
                let idx = self.build(idx.clone(), arr.block);
                let value = self.llvm_builder.build_extract_element(arr.value.into_vector_value(), idx.value.into_int_value(), &self.name());
                Holder { value, block: idx.block }
            },
            PForGather(ndrange, map, gather, _t) => {
                unimplemented!("parallelization is not supported yet")
            },
            IForGather(ndrange, map, gather, _t) => {
                todo!()
            },
            IForReduce(ndrange, initial, reduce, _t) => {
                todo!()
            },
            Uni(operator, expr, _t) => {
                todo!()
            }
            Bin(operator, expr, _t) => {
                todo!()
            }
        }
    }
}