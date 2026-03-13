use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use ftl_rust::PolygonSet;
use memmap2::Mmap;
use ndarray::prelude::*;
use ndarray_npy::ViewNpyExt;
use rayon::prelude::*;

fn ftl_rust(c: &mut Criterion) {
    let mut group = c.benchmark_group("ftl_rust");
    group.sample_size(10);

    let tile_size = 1024;
    let count = 2209;

    let mut path: PathBuf = std::env::current_dir().unwrap();
    path.push("..");
    path.push("data");
    path.push("input_array");
    path.push(format!("test_infile_{}.npy", count));

    let path = path.canonicalize().unwrap();
    println!("reading path {:?}", path);
    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    let data = ArrayView3::<u8>::view_npy(&mmap).unwrap();

    let y_shape = data.shape()[1];
    let x_shape = data.shape()[2];

    let ys: Vec<_> = (0..y_shape).step_by(tile_size).collect();
    let xs: Vec<_> = (0..x_shape).step_by(tile_size).collect();

    group.bench_function(format!("shape {:?}, count {}", data.shape(), count), |b| {
        b.iter(|| {
            let polygon_set = PolygonSet::new(1);
            ys.par_iter().for_each(|&y| {
                let y_max = std::cmp::min(y_shape, y + tile_size);

                xs.par_iter().for_each(|&x| {
                    let x_max = std::cmp::min(x_shape, x + tile_size);

                    let cuboid = data.slice(s![.., y..y_max, x..x_max]).into_dyn();
                    polygon_set._add_tile(cuboid, (0, y, x));
                });
            });
            polygon_set.digest();
            assert!(count == polygon_set.len());
        })
    });
}

criterion_group!(benches, ftl_rust);
criterion_main!(benches);
