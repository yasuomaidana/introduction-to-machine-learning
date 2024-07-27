#[cfg(test)]
mod tests {
    use crate::gram_schmidt;

    #[test]
    fn it_works() {
        let input_vectors: Vec<Vec<f64>> = vec![
            vec![1.0,2.0],
            vec![2.0,0.0],
            vec![3.0,1.0],
        ];
        let basis = gram_schmidt(&input_vectors);
        println!("{:?}", basis);
    }
}