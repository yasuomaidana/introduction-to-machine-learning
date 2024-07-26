use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{pyfunction, pymodule, wrap_pyfunction, Bound, PyResult};
use rayon::prelude::*;

mod test;

#[pyfunction]
fn einstein_mul_rs(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let rows = a.len();
    let cols = b[0].len();
    let mut c = vec![vec![0.0; cols]; rows];

    c.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.par_iter_mut().enumerate().for_each(|(j, c_ij)| {
            *c_ij = a[i]
                .par_iter()
                .zip(b.par_iter().map(|row| row[j]))
                .map(|(a_ij, b_ji)| a_ij * b_ji)
                .sum();
        });
    });

    Ok(c)
}

#[pymodule]
fn einstein_mul_rs_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(einstein_mul_rs, m)?)?;
    Ok(())
}
