mod range;
pub use range::*;
mod stream_ir;
pub use stream_ir::*;
mod optimizer;
pub use optimizer::*;

use std::{fmt::Debug, hash::Hash};

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