use std::{ptr::NonNull, fmt::Debug, cell::RefCell};

pub struct Arena<T, const N: usize = 1024> {
    chunks: RefCell<Vec<(NonNull<[T; N]>, usize)>>,
}

impl<T: Default + Debug, const N: usize> Arena<T, N> {
    fn new() -> Self {
        Self { chunks: RefCell::new(vec![]) }
    }
    unsafe fn push(&self) {
        let mut chunks = self.chunks.borrow_mut();
        chunks.push((Box::<[T; N]>::leak(Box::new([(); N].map(|_| T::default()))).into(), 0));
    }
    fn alloc(&self, x: T) -> &'_ mut T {
        let mut chunks = self.chunks.borrow_mut();
        match chunks.last_mut() {
            Some((space, used)) if *used < N => {
                unsafe { space.as_mut()[*used] = x };
                *used += 1;
                return unsafe { &mut space.as_mut()[*used - 1] }
            },
            _ => {
                drop(chunks);
                unsafe { self.push(); }
                return self.alloc(x);
            }
        }
    }
    fn alloc_ext(&self, mut xs: impl ExactSizeIterator<Item=T>) -> &'_ mut [T] {
        assert!(xs.len() <= N);
        let mut chunks = self.chunks.borrow_mut();
        match chunks.last_mut() {
            Some((space, used)) if *used + xs.len() <= N => {
                let start = *used;
                while let Some(x) = xs.next() {
                    unsafe { space.as_mut()[*used] = x };
                    *used += 1;
                }
                return unsafe { &mut space.as_mut()[start .. *used] }
            },
            _ => {
                drop(chunks);
                unsafe { self.push(); }
                return self.alloc_ext(xs);
            }
        }
    }
}

impl<T, const N: usize> Drop for Arena<T, N> {
    fn drop(&mut self) {
        let mut chunks = self.chunks.borrow_mut();
        for chunk in chunks.iter_mut() {
            drop(unsafe {Box::from_raw(chunk.0.as_ptr())});
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc() {
        let a = Arena::<(u8, bool)>::new();
        let x = (0..20000).map(|_| [(); 20].map(|()| rand::random::<(u8, bool)>())).collect::<Vec<_>>();
        let y = (0..20000).map(|i| a.alloc_ext(x[i].iter().copied())).collect::<Vec<_>>();
        for i in 0..20000 { assert!(x[i] == *y[i]); }
    }
}