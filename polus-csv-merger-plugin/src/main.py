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
    parser.add_argument('--dim', dest='dim', type=str,
                        help='rows or columns', required=True)
    parser.add_argument('--sameRows', dest='sameRows', type=str,
                        help='Only merge csvs if they contain the same number of rows', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    stripExtension = args.stripExtension == 'true'
    logger.info('stripExtension = {}'.format(stripExtension))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    dim = args.dim
    logger.info('dim = {}'.format(dim))
    same_rows = args.sameRows == 'true'
    logger.info('sameRows = {}'.format(stripExtension))
    
    # Start the javabridge with proper java logging
    inpDir_files = [str(f.absolute()) for f in Path(inpDir).iterdir() if f.name.endswith('.csv')]

    # Get the column headers
    logger.info("Getting all unique headers...")
    headers = set([])
    for in_file in inpDir_files:
        with open(in_file,'r') as fr:
            line = fr.readline()
            
            if dim=='columns' and 'file' not in line[0:-1].split(','):
                ValueError('No file columns found in csv: {}'.format(in_file))
            
            headers.update(line[0:-1].split(','))
    if 'file' in headers:
        headers.remove('file')
    headers = list(headers)
    headers.sort()
    headers.insert(0,'file')
    logger.info("Unique headers: {}".format(headers))

    # Generate the line template
    line_template = ','.join([('{' + h + '}') for h in headers]) + '\n'
    line_dict = {key:'NaN' for key in headers}
    
    # Generate the path to the output file
    outPath = str(Path(outDir).joinpath('merged.csv').absolute())
    
    # Merge data
    if dim=='rows':
        logger.info("Merging the data along rows...")
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
                    file_map = {ind:key for ind,key in enumerate(file_map)}

                    for line in in_file:
                        for el,val in enumerate(line.rstrip('\n').split(',')):
                            file_dict[file_map[el]] = val
                        out_file.write(line_template.format(**file_dict))
    elif dim=='columns' and not same_rows:
        logger.info("Merging the data along columns...")
        outPath = str(Path(outDir).joinpath('merged.csv').absolute())
        
        # Load the first csv and generate a dictionary to hold all values
        out_dict = {}
        with open(inpDir_files[0],'r') as in_file:
            file_dict = copy.deepcopy(line_dict)
            
            file_map = in_file.readline().rstrip('\n')
            file_map = file_map.split(',')
            numel = len(file_map)
            file_map = {ind:key for ind,key in zip(range(numel),file_map)}
            
            for line in in_file:
                file_dict = copy.deepcopy(line_dict)
                for el,val in enumerate(line.rstrip('\n').split(',')):
                    file_dict[file_map[el]] = val
                if file_dict['file'] in out_dict.keys():
                    UserWarning('Skipping row for file since it is already in the output file dictionary: {}'.format(file_dict['file']))
                else:
                    out_dict[file_dict['file']] = file_dict
                    
        # Loop through the remaining files and update the output dictionary
        for f in inpDir_files[1:]:
            
            with open(f,'r') as in_file:
                file_dict = copy.deepcopy(line_dict)
                
                file_map = in_file.readline().rstrip('\n')
                file_map = file_map.split(',')
                file_ind = [i for i,v in enumerate(file_map) if v == 'file'][0]
                numel = len(file_map)
                file_map = {ind:key for ind,key in zip(range(numel),file_map)}
                
                for line in in_file:
                    file_dict = copy.deepcopy(line_dict)
                    line_vals = line.rstrip('\n').split(',')
                    for el,val in enumerate(line_vals):
                        if line_vals[file_ind] not in out_dict.keys():
                            out_dict[line_vals[file_ind]] = copy.deepcopy(line_dict)
                            out_dict[line_vals[file_ind]]['file'] = line_vals[file_ind]
                        if el == file_ind:
                            continue
                        if out_dict[line_vals[file_ind]][file_map[el]] != 'NaN':
                            Warning('Skipping duplicate value ({}) found in {}'.format(val,in_file))
                        else:
                            out_dict[line_vals[file_ind]][file_map[el]] = val
                            
        # Write the output file
        with open(outPath,'w') as out_file:
            out_file.write(','.join(headers) + '\n')
            
            for val in out_dict.values():
                out_file.write(line_template.format(**val))
    
    elif dim=='columns':
        pass