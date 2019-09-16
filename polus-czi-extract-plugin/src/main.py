import argparse
import time
from pathlib import Path
from bfio import czi2tif
import javabridge as jutil

def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Extract individual fields of view from a czi file.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)

    print('Arguments:')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print('input_dir = {}'.format(input_dir))
    print('output_dir = {}'.format(output_dir))

    print(' ')
    
    print('Initializing the javabridge...')
    JARS = jutil.JARS
    bfjar = jutil.JARS[0][0:jutil.JARS[0].find("javabridge")]
    JARS.append(bfjar + "bioformats/jars/loci_tools.jar")
    jutil.start_vm(class_path=JARS)

    print('Extracting tiffs and saving as ome.tif...')
    files = [f for f in Path(input_dir).iterdir() if f.is_file() and f.suffix=='.czi']
    if not files:
        ValueError('No CZI files found.')
    
    for file in files:
        start_time = time.time()
        print('Starting extraction from ' + str(file.absolute()) + '...')
        czi2tif.write_ome_tiffs(str(file.absolute()),output_dir)
        print('Finished in {}s!'.format(time.time()-start_time))
        
    print('Finished extracting files. Closing the javabridge and exiting...')
    jutil.kill_vm()

if __name__ == "__main__":
    main()