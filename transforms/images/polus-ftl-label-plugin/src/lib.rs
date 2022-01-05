mod polygons;

use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;

pub use polygons::Polygon;
pub use polygons::PolygonSet;

#[pyfunction]
pub fn extract_tile<'py>(
    py: Python<'py>,
    polygon_set: &PolygonSet,
    coordinates: (usize, usize, usize, usize, usize, usize),
) -> &'py PyArrayDyn<usize> {
    let tile = polygon_set._extract_tile(coordinates);
    tile.into_pyarray(py)
}

/// Generates a Python-class for interfacing with Python.
#[pymodule]
fn ftl_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PolygonSet>()?;
    m.add_function(wrap_pyfunction!(extract_tile, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::prelude::*;
    use ndarray_npy::read_npy;
    use rayon::prelude::*;

    use crate::PolygonSet;

    fn read_array(count: usize) -> Array3<u8> {
        let mut path: PathBuf = std::env::current_dir().unwrap();
        path.push("..");
        path.push("data");
        path.push("input_array");
        path.push(format!("test_infile_{}.npy", count));
        println!("reading path {:?}", path);

        read_npy(path).unwrap()
    }

    #[test]
    fn test_array() {
        let tile_size = 1024;
        let count = 63;
        let data = read_array(count);
        let polygon_set = PolygonSet::new(1);

        let n_rows = data.shape()[1];
        let n_cols = data.shape()[2];

        let ys: Vec<_> = (0..n_rows).step_by(tile_size).collect();
        let xs: Vec<_> = (0..n_cols).step_by(tile_size).collect();

        ys.par_iter().for_each(|&y| {
            let y_max = std::cmp::min(n_rows, y + tile_size);

            xs.par_iter().for_each(|&x| {
                let x_max = std::cmp::min(n_cols, x + tile_size);
                let tile = data.slice(s![.., y..y_max, x..x_max]).into_dyn();
                polygon_set._add_tile(tile, (0, y, x));
            });
        });

        polygon_set.digest();

        assert_eq!(polygon_set.len(), count, "wrong number of polygons");
    }
}
