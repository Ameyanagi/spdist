use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::ndarray::Zip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

mod rust_lib;

#[pymodule]
/// A collection of functions for calculating distances between 2 curves
fn spdist<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn spdist<'py>(
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        x_ref: PyReadonlyArray1<'py, f64>,
        y_ref: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let distance = rust_lib::calc_distance_spdist(
            x.as_array(),
            y.as_array(),
            x_ref.as_array(),
            y_ref.as_array(),
        );

        match distance {
            Ok(distance) => Ok(distance),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    #[pyfn(m)]
    fn spdist_vector<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        x_ref: PyReadonlyArray1<'py, f64>,
        y_ref: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let distance = rust_lib::calc_distance_spdist_vector(
            x.as_array(),
            y.as_array(),
            x_ref.as_array(),
            y_ref.as_array(),
        );

        match distance {
            Ok(distance) => Ok(distance.into_pyarray(py).to_owned()),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    #[pyfn(m)]
    fn squared_spdist<'py>(
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        x_ref: PyReadonlyArray1<'py, f64>,
        y_ref: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let distance = rust_lib::calc_squared_distance_spdist(
            x.as_array(),
            y.as_array(),
            x_ref.as_array(),
            y_ref.as_array(),
        );

        match distance {
            Ok(distance) => Ok(distance),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    #[pyfn(m)]
    fn squared_spdist_vector<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        x_ref: PyReadonlyArray1<'py, f64>,
        y_ref: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let distance = rust_lib::calc_squared_distance_spdist_vector(
            x.as_array(),
            y.as_array(),
            x_ref.as_array(),
            y_ref.as_array(),
        );

        match distance {
            Ok(distance) => Ok(distance.into_pyarray(py).to_owned()),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }
    Ok(())
}
