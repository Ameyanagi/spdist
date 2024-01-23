use std::error::Error;
use std::fmt::Display;

use ndarray::parallel::prelude::IntoParallelIterator;
use numpy::ndarray::{parallel, s, ArrayBase, Ix1, OwnedRepr, ViewRepr, Zip};
use rayon::prelude::*;

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

pub fn calc_distance<'a>(
    x: ArrayBase<ViewRepr<&'a f64>, Ix1>,
    y: ArrayBase<ViewRepr<&'a f64>, Ix1>,
    x_ref: ArrayBase<ViewRepr<&'a f64>, Ix1>,
    y_ref: ArrayBase<ViewRepr<&'a f64>, Ix1>,
) -> Result<f64, SpdistError> {
    if x.len() != y.len() {
        return Err(SpdistError::VectorSizeMismatch);
    }

    if x_ref.len() != y_ref.len() {
        return Err(SpdistError::VectorSizeMismatch);
    }

    let distance = Zip::from(&x)
        .and(&y)
        .into_par_iter()
        .map(|(x, y)| {
            Zip::from(&x_ref.slice(s![..-1]))
                .and(&y_ref.slice(s![..-1]))
                .and(&x_ref.slice(s![1..]))
                .and(&y_ref.slice(s![1..]))
                .into_par_iter()
                .map(|(x_ref, y_ref, x_ref_next, y_ref_next)| -> f64 {
                    // return point to point distance
                    if (x_ref == x_ref_next) && (y_ref == y_ref_next) {
                        return ((x - x_ref).powi(2) + (y - y_ref).powi(2)).sqrt();
                    }
                    // return point to line distance
                    // https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
                    ((x_ref_next - x_ref) * (y_ref - y) - (x_ref - x) * (y_ref_next - y_ref)).abs()
                        / ((x_ref_next - x_ref).powi(2) + (y_ref_next - y_ref).powi(2)).sqrt()
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                .unwrap_or(0.0)
        })
        .reduce(|| 0.0f64, |acc, x| acc + x)
        / (x.len() as f64);

    Ok(distance)
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::{self, abs_diff_eq, assert_relative_eq};
    use numpy::ndarray::Array1;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_calc_distance() -> Result<(), Box<dyn Error>> {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![1.0, 1.0];

        let x_ref: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_ref: Vec<f64> = vec![1.0, 2.0, 3.0];

        let x = Array1::from(x);
        let y = Array1::from(y);
        let x_ref = Array1::from(x_ref);
        let y_ref = Array1::from(y_ref);

        let expected = (1.0 / 2.0 as f64).sqrt() / (x.len() as f64);

        let distance = calc_distance(x.view(), y.view(), x_ref.view(), y_ref.view())?;

        assert_relative_eq!(expected, distance, epsilon = TOL);

        Ok(())
    }

    #[test]
    fn test_calc_distance_duplicate_ref() -> Result<(), Box<dyn Error>> {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![1.0, 1.0];

        let x_ref: Vec<f64> = vec![1.0, 2.0, 2.0, 3.0];
        let y_ref: Vec<f64> = vec![1.0, 2.0, 2.0, 3.0];

        let x = Array1::from(x);
        let y = Array1::from(y);
        let x_ref = Array1::from(x_ref);
        let y_ref = Array1::from(y_ref);

        let expected = (1.0 / 2.0 as f64).sqrt() / (x.len() as f64);

        let distance = calc_distance(x.view(), y.view(), x_ref.view(), y_ref.view())?;

        assert_relative_eq!(expected, distance, epsilon = TOL);

        Ok(())
    }

    #[test]
    fn test_calc_distance_duplicate_input() -> Result<(), Box<dyn Error>> {
        let x: Vec<f64> = vec![1.0, 2.0, 2.0];
        let y: Vec<f64> = vec![1.0, 1.0, 1.0];

        let x_ref: Vec<f64> = vec![1.0, 2.0, 2.0, 3.0];
        let y_ref: Vec<f64> = vec![1.0, 2.0, 2.0, 3.0];

        let x = Array1::from(x);
        let y = Array1::from(y);
        let x_ref = Array1::from(x_ref);
        let y_ref = Array1::from(y_ref);

        let expected = (1.0 / 2.0 as f64).sqrt() / (x.len() as f64) * 2.0;

        let distance = calc_distance(x.view(), y.view(), x_ref.view(), y_ref.view())?;

        assert_relative_eq!(expected, distance, epsilon = TOL);

        Ok(())
    }
}
