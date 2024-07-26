
#[cfg(test)]
mod tests {
    use crate::{einstein_mul_rs};
    #[test]
    fn it_works() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = einstein_mul_rs(a, b);

        match result {
            Ok(c) => {
                assert_eq!(c, vec![vec![7.0, 10.0], vec![15.0, 22.0]]);
            },
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        }
    }
}