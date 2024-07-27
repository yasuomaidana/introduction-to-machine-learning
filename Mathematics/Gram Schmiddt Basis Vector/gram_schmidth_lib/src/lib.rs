mod lib_test;

#[derive(Debug)]
pub struct Basis{
    pub rank: usize,
    pub basis: Vec<Vec<f64>>,
}

fn dot_product(v1:&Vec<f64>, v2:&Vec<f64>) -> f64{
    v1.iter().zip(v2.iter())
        .map(|(x, y)|  x * y)
        .sum()
}

fn norm(v:&Vec<f64>) -> f64{
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn norm_vector(v:&Vec<f64>) -> Vec<f64>{
    let norm = norm(v);
    v.iter().map(|x| x / norm).collect()
}

/// Computes the projection of vector `v1` onto vector `v2`.
fn projection_of_vector(v1:&Vec<f64>, v2:&Vec<f64>) -> Vec<f64>{
    let dot = dot_product(v1, v2);
    norm_vector(&v2).iter().map(|x| x * dot).collect()
}

pub fn gram_schmidt(vectors:&Vec<Vec<f64>>) -> Basis {
    let mut basis_vectors:Vec<Vec<f64>> = Vec::new();
    let mut rank:usize = 0;
    for vector in vectors.iter(){
        if rank > 0 {
            let mut new_vector = vector.clone();
            for i in 0..rank{
                let projection = projection_of_vector(&vector, &basis_vectors[i]);
                new_vector = new_vector.iter().zip(projection.iter()).map(|(x, y)| x - y).collect();
            }
            if norm(&new_vector) <= 1e-6{
                break;
            }

            basis_vectors.push(norm_vector(&new_vector));
        }else{
            basis_vectors.push(norm_vector(&vector));
        }
        rank += 1;
    }

    Basis{rank, basis:basis_vectors}
}

