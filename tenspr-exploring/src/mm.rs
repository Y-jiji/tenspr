#[derive(Clone)]
pub enum Expr {
    Elem(usize),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

impl std::fmt::Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Elem(u) => write!(f, "{u:04}"),
            Expr::Add(a, b) => write!(f, "{{ {a:?} + {b:?} }}"),
            Expr::Mul(a, b) => write!(f, "({a:?} * {b:?})"),
        }
    }
}

impl std::ops::Add<Expr> for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul<Expr> for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

pub fn matmul(a: Vec<Vec<Expr>>, b: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    let mut c = vec![vec![Expr::Elem(0); b.get(0).map(|i| i.len()).unwrap_or(0)]; a.len()];
    for i in 0..a.len() {
        for j in 0..b.len() {
            for k in 0..b[0].len() {
                c[i][k] = c[i][k].clone() + a[i][j].clone() * b[j][k].clone();
            }
        }
    }
    return c;
}

pub fn matmul_tiled(a: Vec<Vec<Expr>>, b: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    const T: usize = 2usize;
    let mut c = vec![vec![Expr::Elem(0); b.get(0).map(|i| i.len()).unwrap_or(0)]; a.len()];
    for i in 0..(a.len() + T - 1) / T {
        for k in 0..(b[0].len() + T - 1) / T {
            for j in 0..(b.len() + T - 1) / T {
                for ti in i * T..((i + 1) * T).min(a.len()) {
                    for tk in k * T..((k + 1) * T).min(b[0].len()) {
                        for tj in j * T..((j + 1) * T).min(b.len()) {
                            c[ti][tk] = c[ti][tk].clone() + a[ti][tj].clone() * b[tj][tk].clone();
                        }
                    }
                }
            }
        }
    }
    return c;
}

#[cfg(test)]
mod tests {
    use super::{matmul, matmul_tiled};

    #[test]
    fn test_matmul_tiled() {
        let a = (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| super::Expr::Elem(i * 10 + j + 000))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let b = (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| super::Expr::Elem(i * 10 + j + 100))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (i, row) in matmul(a.clone(), b.clone()).into_iter().enumerate() {
            for (j, col) in row.into_iter().enumerate() {
                println!("{i} {j} {col:?}");
            }
        }
        println!("===================================");
        for (i, row) in matmul_tiled(a.clone(), b.clone()).into_iter().enumerate() {
            for (j, col) in row.into_iter().enumerate() {
                println!("{i} {j} {col:?}");
            }
        }
    }
}