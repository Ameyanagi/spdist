// use pyo3::prelude::*;
use std::error::Error;
use std::fmt::Display;

use numpy::ndarray::Zip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[derive(Debug)]
pub enum SpdistError {
    VectorSizeMismatch,
}

impl Display for SpdistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpdistError::VectorSizeMismatch => write!(f, "Vector size VectorSizeMismatch"),
        }
    }
}

impl Error for SpdistError {}

pub fn calc_distance(
    x: Vec<f64>,
    y: Vec<f64>,
    x_ref: Vec<f64>,
    y_ref: Vec<f64>,
) -> Result<f64, SpdistError> {
    if x.len() != y.len() {
        return Err(SpdistError::VectorSizeMismatch);
    }

    if x_ref.len() != y.len() {
        return Err(SpdistError::VectorSizeMismatch);
    }

    let dist = x
        .iter()
        .zip(y.iter())
        .map(|(x, y)| {
            x_ref
                .iter()
                .zip(y_ref.iter())
                .map(|(x_tmp, y_tmp)| ((x - x_tmp).powi(2) + (y - y_tmp).powi(2)).sqrt())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        })
        .fold(0.0, |acc, x| acc + x)
        / (x.len() as f64);

    Ok(dist)
}

#[pymodule]
fn spdist<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> &'py PyArray1<f64> {
        let x = x.as_array();
        let y = y.as_array();

        let z = Zip::from(x).and(&y).par_map_collect(|x, y| x + y);
        z.into_pyarray(py)
    }
    Ok(())
}
