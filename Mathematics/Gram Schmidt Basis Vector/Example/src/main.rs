use gram_schmidt_lib::gram_schmidt;
fn transpose(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_rows = matrix.len();
    let num_cols = matrix[0].len();

    let mut transposed: Vec<Vec<f64>> = vec![vec![0.0; num_rows]; num_cols];

    for i in 0..num_rows {
        for j in 0..num_cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

fn main() {
     // Example input vectors
    let input_vectors: Vec<Vec<f64>> = vec![
       vec![1.0,0.0,2.0,6.0],
       vec![0.0, 1.0, 8.0, 2.0],
       vec![2.0, 8.0,3.0,1.0],
       vec![1.0, -6.0,2.0,3.0],
   ];

   let input_vectors = transpose(input_vectors.clone());

   for row in input_vectors.iter(){
      for val in row.iter(){
          print!("{:?} ,", val);
      }
        println!();
   }

   // Call the gram_schmidt function from the library
   let basis = gram_schmidt(&input_vectors);

   println!("Rank of the basis: {:?}", basis.rank);
   println!("Orthogonal Basis Vectors:");

   for row in transpose(basis.basis_vectors).iter(){
      for val in row.iter(){
          print!("{:?} ,", val);
      }
        println!();
   }

   // for vec in basis.basis_vectors {
   //     println!("{:?}", vec);
   // }
}
