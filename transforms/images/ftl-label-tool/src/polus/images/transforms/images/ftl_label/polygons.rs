use std::cmp::max;
use std::cmp::min;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::RwLock;

// In numpy 0.22, ndarray types are re-exported via numpy::ndarray
use numpy::ndarray::{ArrayD, ArrayViewD, Array3};
use numpy::ndarray::prelude::*;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use rayon::prelude::*;

type Slice = (usize, (usize, (usize, usize))); // (z, (y, (x_min, x_max)))
type PolyVec = Vec<Arc<Polygon>>;

fn do_slices_overlap(left: Slice, right: Slice, connectivity: u8) -> bool {
    let (left_z, (left_y, (left_start, left_stop))) = left;
    let (right_z, (right_y, (right_start, right_stop))) = right;

    if (left_start > right_stop) || (right_start > left_stop) {
        return false;
    }

    let z_diff = left_z.abs_diff(right_z);
    if z_diff > 1 {
        return false;
    }

    let y_diff = left_y.abs_diff(right_y);
    if y_diff > 1 {
        return false;
    }

    if z_diff == 0 && y_diff == 0 {
        (left_start == right_stop) || (right_start == left_stop)
    } else if y_diff == 1 && z_diff == 1 {
        if connectivity == 3 {
            (left_start <= right_stop) || (right_start <= left_stop)
        } else {
            (left_start < right_stop) || (right_start < left_stop)
        }
    } else if connectivity == 1 {
        (left_start < right_stop) || (right_start < left_stop)
    } else {
        (left_start <= right_stop) || (right_start <= left_stop)
    }
}

/// A `Polygon` represents a single connected object to be labelled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polygon {
    connectivity: u8,
    slices: Vec<Slice>,
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
    z_min: usize,
    z_max: usize,
}

impl Polygon {
    pub fn new(connectivity: u8, slices: Vec<Slice>) -> Self {
        assert!(!slices.is_empty(), "Cannot create Polygon without any slices.");

        let (zs, rest): (Vec<_>, Vec<_>) = slices.iter().copied().unzip();
        let z_min = *zs.iter().min().unwrap();
        let z_max = zs.into_iter().max().unwrap() + 1;

        let (ys, rest): (Vec<_>, Vec<_>) = rest.into_iter().unzip();
        let y_min = *ys.iter().min().unwrap();
        let y_max = ys.into_iter().max().unwrap() + 1;

        let (starts, stops): (Vec<_>, Vec<_>) = rest.into_iter().unzip();
        let x_min = starts.into_iter().min().unwrap();
        let x_max = stops.into_iter().max().unwrap();

        Polygon { connectivity, slices, x_min, x_max, y_min, y_max, z_min, z_max }
    }

    pub fn is_empty(&self) -> bool { self.slices.is_empty() }
    pub fn len(&self) -> usize { self.slices.len() }

    pub fn bbox_overlap(&self, other: &Self) -> bool {
        self.x_min <= other.x_max
            && self.y_min <= other.y_max
            && self.z_min <= other.z_max
            && self.x_max >= other.x_min
            && self.y_max >= other.y_min
            && self.z_max >= other.z_min
    }

    pub fn boundary_connects(&self, other: &Self) -> bool {
        if self.bbox_overlap(other) {
            self.slices.par_iter().any(|&left| {
                other.slices.par_iter()
                    .any(|&right| do_slices_overlap(left, right, self.connectivity))
            })
        } else {
            false
        }
    }

    pub fn absorb(&mut self, other: &mut Self) {
        self.x_min = min(self.x_min, other.x_min);
        self.y_min = min(self.y_min, other.y_min);
        self.z_min = min(self.z_min, other.z_min);
        self.x_max = max(self.x_max, other.x_max);
        self.y_max = max(self.y_max, other.y_max);
        self.z_max = max(self.z_max, other.z_max);
        self.slices.extend(other.slices.drain(..));
    }
}

fn bft_partition(polygons: &mut Vec<Polygon>, connectivity: u8) -> Vec<Polygon> {
    let mut merged: Vec<Polygon> = Vec::new();

    polygons.iter_mut().for_each(|target| {
        merged.iter_mut()
            .filter(|c| !c.is_empty())
            .for_each(|candidate| {
                if target.boundary_connects(candidate) {
                    target.absorb(candidate);
                }
            });
        merged.push(Polygon {
            connectivity,
            slices: target.slices.drain(..).collect(),
            x_min: target.x_min, y_min: target.y_min, z_min: target.z_min,
            x_max: target.x_max, y_max: target.y_max, z_max: target.z_max,
        });
    });

    merged.iter_mut()
        .filter(|p| !p.is_empty())
        .map(|p| Polygon {
            connectivity,
            slices: p.slices.drain(..).collect(),
            x_min: p.x_min, y_min: p.y_min, z_min: p.z_min,
            x_max: p.x_max, y_max: p.y_max, z_max: p.z_max,
        })
        .collect()
}

impl Ord for Polygon {
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap() }
}

impl PartialOrd for Polygon {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.z_min.cmp(&other.z_min) {
            Ordering::Equal => match self.y_min.cmp(&other.y_min) {
                Ordering::Equal => match self.x_min.cmp(&other.x_min) {
                    Ordering::Equal => match self.z_max.cmp(&other.z_max) {
                        Ordering::Equal => match self.y_max.cmp(&other.y_max) {
                            Ordering::Equal => Some(self.x_max.cmp(&other.x_max)),
                            other => Some(other),
                        },
                        other => Some(other),
                    },
                    other => Some(other),
                },
                other => Some(other),
            },
            other => Some(other),
        }
    }
}

/// A `PolygonSet` handles and maintains `Polygons` in an image.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PolygonSet {
    connectivity: u8,
    polygons: Arc<RwLock<PolyVec>>,
}

#[pymethods]
impl PolygonSet {
    #[new]
    pub fn new(connectivity: u8) -> Self {
        PolygonSet { connectivity, polygons: Arc::new(RwLock::new(Vec::new())) }
    }

    pub fn is_empty(&self) -> bool { self.polygons.read().unwrap().is_empty() }
    pub fn len(&self) -> usize { self.polygons.read().unwrap().len() }

    pub fn add_tile(&self, tile: PyReadonlyArrayDyn<'_, u8>, top_left_point: (usize, usize, usize)) {
        self._add_tile(tile.as_array(), top_left_point)
    }

    pub fn digest(&self) {
        let mut polygons = self.polygons.write().unwrap()
            .drain(..)
            .map(|p| Polygon {
                connectivity: p.connectivity,
                slices: p.slices.iter().copied().collect(),
                x_min: p.x_min, y_min: p.y_min, z_min: p.z_min,
                x_max: p.x_max, y_max: p.y_max, z_max: p.z_max,
            })
            .collect();
        let mut polygons = bft_partition(&mut polygons, self.connectivity);
        polygons.sort();
        self.polygons.write().unwrap()
            .extend(polygons.drain(..).map(Arc::new));
    }
}

impl PolygonSet {
    pub fn _add_tile(&self, tile: ArrayViewD<'_, u8>, top_left_point: (usize, usize, usize)) {
        let (z_min, y_min, x_min) = top_left_point;

        let mut slices = tile
            .outer_iter().into_par_iter().enumerate()
            .flat_map(|(z, plane)| {
                let mut slices = plane
                    .outer_iter().into_par_iter().enumerate()
                    .flat_map(|(y, row)| {
                        let runs: Vec<_> = row.iter()
                            .chain([0].iter())
                            .zip([0].iter().chain(row.iter()))
                            .enumerate()
                            .filter(|(_, (&a, &b))| a != b)
                            .map(|(i, _)| i)
                            .collect();

                        if runs.is_empty() {
                            Vec::new()
                        } else {
                            let starts: Vec<_> = runs.iter().step_by(2).cloned().collect();
                            let ends: Vec<_> = runs.into_iter().skip(1).step_by(2).collect();
                            starts.into_par_iter().zip(ends.into_par_iter())
                                .map(|(start, stop)| Polygon::new(
                                    self.connectivity,
                                    vec![(z + z_min, (y + y_min, (start + x_min, stop + x_min)))],
                                ))
                                .collect::<Vec<_>>()
                        }
                    })
                    .collect::<Vec<_>>();
                bft_partition(&mut slices, self.connectivity)
            })
            .collect::<Vec<_>>();

        let mut polygons = bft_partition(&mut slices, self.connectivity);
        self.polygons.write().unwrap()
            .extend(polygons.drain(..).map(Arc::new));
    }

    pub fn _extract_tile(&self, coordinates: (usize, usize, usize, usize, usize, usize)) -> ArrayD<usize> {
        let (z_min, z_max, y_min, y_max, x_min, x_max) = coordinates;

        let tile_polygon = Polygon::new(
            self.connectivity,
            vec![
                (z_min, (y_min, (x_min, x_max))),
                (z_max, (y_max, (x_min, x_max))),
            ],
        );

        let mut tile: Array3<usize> = Array3::zeros((z_max - z_min, y_max - y_min, x_max - x_min));

        self.polygons.read().unwrap().iter().enumerate()
            .for_each(|(i, polygon)| {
                if tile_polygon.bbox_overlap(polygon) {
                    polygon.slices.iter().copied()
                        .for_each(|(z, (y, (start, stop)))| {
                            if z_min <= z && z < z_max && y_min <= y && y < y_max {
                                let start = max(x_min, start);
                                let stop = min(x_max, stop);
                                let mut section = tile.slice_mut(s![
                                    z - z_min, y - y_min, (start - x_min)..(stop - x_min)
                                ]);
                                section.fill(i + 1);
                            }
                        });
                }
            });

        tile.into_dyn()
    }
}
