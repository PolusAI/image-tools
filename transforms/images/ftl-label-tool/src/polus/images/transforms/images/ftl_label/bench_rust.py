import time
from pathlib import Path

from ftl_rust import PolygonSet


def bench_rust():
    count = 2209
    infile = Path(f'../../data/input_array/test_infile_{count}.ome.tif').resolve()
    outfile = Path(f'../../data/input_array/test_outfile_{count}.ome.tif').resolve()
    polygon_set = PolygonSet(connectivity=1)
    
    start = time.time()
    polygon_set.read_from(infile)
    end = time.time()
    print(f'took {end - start:.3f} seconds to read and digest...')

    assert count == len(polygon_set), f'found {len(polygon_set)} objects instead of {count}.'

    polygon_set.write_to(outfile)
    print(f'took {time.time() - end:.3f} seconds to write...')
    return


if __name__ == '__main__':
    bench_rust()
