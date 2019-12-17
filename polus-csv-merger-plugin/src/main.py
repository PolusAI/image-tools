import argparse, logging, os, glob, copy
from pathlib import Path

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Merge all csv files in a csv collection into a single csv file.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--stripExtension', dest='stripExtension', type=str,
                        help='Should csv be removed from the filename when indicating which file a row in a csv file came from?', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output csv file', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    stripExtension = args.stripExtension == 'true'
    logger.info('stripExtension = {}'.format(stripExtension))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Start the javabridge with proper java logging
    inpDir_files = glob.glob(os.path.join(inpDir, '*.csv'))

    # Get the column headers
    logger.info("Getting all unique headers...")
    headers = set([])
    for in_file in inpDir_files:
        with open(in_file,'r') as fr:
            line = fr.readline()
            headers.update(line[0:-1].split(','))
    headers = list(headers)
    headers.insert(0,'file')
    logger.info("Unique headers: {}".format(headers))

    # Generate the line template
    line_template = ','.join([('{' + h + '}') for h in headers]) + '\n'
    line_dict = {key:'NaN' for key in headers}
    
    # Merge data
    logger.info("Merging the data...")
    outPath = str(Path(outDir).joinpath('merged.csv').absolute())
    with open(outPath,'w') as out_file:
        out_file.write(','.join(headers) + '\n')
        for f in inpDir_files:
            file_dict = copy.deepcopy(line_dict)
            if stripExtension:
                file_dict['file'] = str(Path(f).name).replace('.csv','')
            else:
                file_dict['file'] = str(Path(f).name)
            logger.info("Merging file: {}".format(file_dict['file']))

            with open(f,'r') as in_file:
                file_map = in_file.readline().rstrip('\n')
                file_map = file_map.split(',')
                numel = len(file_map)
                file_map = {ind:key for ind,key in zip(range(numel),file_map)}

                for line in in_file:
                    for el,val in zip(range(numel),line.rstrip('\n').split(',')):
                        file_dict[file_map[el]] = val
                    out_file.write(line_template.format(**file_dict))
    