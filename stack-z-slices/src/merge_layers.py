from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
from pathlib import Path
from utils import _parse_files_xy,_parse_files_p,_parse_regex
import argparse

def _merge_layers(input_dir,input_files,output_dir,output_file):
    zs = [z for z in input_files.keys()] # sorted list of filenames by z-value
    
    # Initialize the output file
    br = BioReader(str(Path(input_dir).joinpath(input_files[zs[0]][0]).absolute()))
    bw = BioWriter(str(Path(output_dir).joinpath(output_file).absolute()),metadata=br.read_metadata())
    bw.num_z(Z = len(zs))
    del br
    
    # Load each image and save to the volume file
    for z,i in zip(zs,range(len(zs))):
        br = BioReader(str(Path(input_dir).joinpath(input_files[z][0]).absolute()))
        bw.write_image(br.read_image(),Z=[i,i+1])
        del br
    
    # Close the output image and delete
    bw.close_image()
    del bw

if __name__ == "__main__":
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    parser = argparse.ArgumentParser(prog='merge_layers', description='Merge images into a single volume.')
    
    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--regex', dest='regex', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--P', dest='P', type=str,
                        help='P image position', required=False)
    parser.add_argument('--X', dest='X', type=str,
                        help='X image position', required=False)
    parser.add_argument('--Y', dest='Y', type=str,
                        help='Y image position', required=False)
    parser.add_argument('--T', dest='T', type=str,
                        help='T image position', required=True)
    parser.add_argument('--C', dest='C', type=str,
                        help='C image position', required=True)
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    regex = args.regex
    if args.P:
        P = int(args.P)
        X = None
        Y = None
    else:
        P = None
        X = int(args.X)
        Y = int(args.Y)
    C = int(args.C)
    T = int(args.T)
    
    # Parse the regex
    regex,variables = _parse_regex(regex)
    
    # Parse files based on regex
    if 'p' not in variables:
        files = _parse_files_xy(input_dir,regex,variables)
    else:
        files = _parse_files_p(input_dir,regex,variables)
    
    # Stack images based on positioning variable used
    if P == None:
        zs = [z for z in files[T][C][X][Y].keys()]
        zs.sort()
        _merge_layers(input_dir,files[T][C][X][Y],output_dir,files[T][C][X][Y][zs[0]][0])
    else:
        zs = [z for z in files[T][C][P].keys()]
        zs.sort()
        _merge_layers(input_dir,files[T][C][P],output_dir,files[T][C][P][zs[0]][0])
        
    jutil.kill_vm()