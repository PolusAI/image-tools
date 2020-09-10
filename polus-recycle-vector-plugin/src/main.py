import argparse, time, os, logging, re, filepattern
from pathlib import Path

def close_vectors(vectors):
    if isinstance(vectors,dict):
        for key in vectors:
            close_vectors(vectors[key])
    else:
        vectors.close()
        
def main():
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Extract individual fields of view from a czi file.')

    parser.add_argument('--stitchDir', dest='stitch_dir', type=str,
                        help='Stitching vector to recycle', required=True)
    parser.add_argument('--collectionDir', dest='collection_dir', type=str,
                        help='Image collection to place in new stitching vector', required=True)
    parser.add_argument('--stitchRegex', dest='stitch_regex', type=str,
                        help='Stitching vector regular expression', required=True)
    parser.add_argument('--collectionRegex', dest='collection_regex', type=str,
                        help='Image collection regular expression', required=True)
    parser.add_argument('--groupBy', dest='group_by', type=str,
                        help='Variables to group within a single stitching vector', required=False)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The directory in which to save stitching vectors.', required=True)

    # Get the arguments
    args = parser.parse_args()
    stitch_dir = args.stitch_dir
    collection_dir = args.collection_dir
    stitch_regex = args.stitch_regex
    collection_regex = args.collection_regex
    group_by = args.group_by
    output_dir = args.output_dir
    logger.info('stitch_dir = {}'.format(stitch_dir))
    logger.info('collection_dir = {}'.format(collection_dir))
    logger.info('stitch_regex = {}'.format(stitch_regex))
    logger.info('collection_regex = {}'.format(collection_regex))
    logger.info('output_dir = {}'.format(output_dir))
    
    # Process group_by variable
    if group_by==None:
        group_by = ''
    for v in 'xyp':
        if v not in group_by:
            group_by += v
    group_by = group_by.lower()
    logger.info('group_by = {}'.format(group_by))
    
    # Parse files in the image collection
    fp = filepattern.FilePattern(collection_dir,collection_regex)
    
    # Loop through the stitching vectors
    vectors = [v for v in Path(stitch_dir).iterdir() if Path(v).name.startswith('img-global-positions')]
    vector_count = 1
    for vector in vectors:
        
        logger.info("Processing vector: {}".format(str(vector.absolute())))
        
        # Parse the stitching vector
        sp = filepattern.VectorPattern(str(vector.absolute()),stitch_regex)
        if sp.variables == None:
            ValueError('Stitching vector pattern must contain variables.')
        
        # Grouping variables for files in the image collection
        file_groups = [v for v in sp.var_order if v not in group_by]
        vector_groups = [v for v in file_groups if v not in 'xyp']
        
        # Vector output dictionary
        vector_dict = {}
        
        # Loop through lines in the stitching vector, generate new vectors
        for v in sp.iterate():
            variables = {key.upper():value for key,value in v[0].items() if key in group_by}
            file_matches = fp.get_matching(**variables)
            
            for f in file_matches:
                
                # Get the file writer, create it if it doesn't exist
                temp_dict = vector_dict
                for key in vector_groups:
                    if f[key] not in temp_dict.keys():
                        if vector_groups[-1] != key:
                            temp_dict[f[key]] = {}
                        else:
                            fname = "img-global-positions-{}.txt".format(vector_count)
                            vector_count += 1
                            out_vars = {key:f[key] for key in vector_groups}
                            logger.info("Creating vector ({}) for variables: {}".format(fname,out_vars))
                            temp_dict[f[key]] = open(str(Path(output_dir).joinpath(fname).absolute()),'w')
                    temp_dict = temp_dict[f[key]]
                
                # If the only grouping variables are positional (xyp), then create an output file
                fw = temp_dict
                
                fw.write("file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n".format(Path(f['file']).name,
                                                                                            v[0]['correlation'],
                                                                                            v[0]['posX'],
                                                                                            v[0]['posY'],
                                                                                            v[0]['gridX'],
                                                                                            v[0]['gridY']))
        
        # Close all open stitching vectors
        close_vectors(vector_dict)

    logger.info("Plugin completed all operations!")

if __name__ == "__main__":
    main()