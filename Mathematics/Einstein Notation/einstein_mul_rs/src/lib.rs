use pyo3::{Bound, pyfunction, pymodule, PyResult, wrap_pyfunction};
use pyo3::prelude::{PyModule, PyModuleMethods};
use rayon::prelude::*;

mod test;

#[pyfunction]
fn einstein_mul_rs(a:Vec<Vec<f64>>,b:Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>>{
    let rows = a.len();
    let cols = b[0].len();
    let mut c = vec![vec![0.0; cols]; rows];

    c.par_iter_mut().enumerate().
        for_each(|(i, row)| {
        for j in 0..cols {
            for k in 0..b.len() {
                row[j] += a[i][k] * b[k][j];
            }
        }
    });

    Ok(c)
}

#[pymodule]
fn einstein_mul_rs_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(einstein_mul_rs, m)?)?;
    Ok(())
}
