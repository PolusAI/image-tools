import argparse, logging, math
from pathlib import Path

def get_number(s):
    """ Check that s is number
    
    If s is a number, returns it as a float. If not, returns s without modification.

    Inputs:
        s - An input string or number
    Outputs:
        value - Either float(s) or False if s cannot be cast to float
    """
    try:
        return float(s)
    except ValueError:
        return s
    
def count(data_list,data_dict):
    if 'count' in data_dict.keys():
        return
    data_dict['count'] = len(data_list)
    return data_dict
    
def minval(data_list,data_dict):
    if 'min' in data_dict.keys():
        return
    data_dict['min'] = min(data_list)
    return data_dict

def maxval(data_list,data_dict):
    if 'max' in data_dict.keys():
        return
    data_dict['max'] = max(data_list)
    return data_dict
    
def mean(data_list,data_dict):
    if 'mean' in data_dict.keys():
        return
    count(data_list,data_dict)
    data_dict['mean'] = sum(data_list)/data_dict['count']
    return data_dict
    
def var(data_list,data_dict):
    if 'var' in data_dict.keys():
        return
    mean(data_list,data_dict)
    data_dict['var'] = sum([x**2 for x in data_list])/data_dict['count'] - data_dict['mean']**2
    return data_dict

def median(data_list,data_dict):
    if 'median' in data_dict.keys():
        return
    count(data_list,data_dict)
    data_list.sort()
    data_dict['median'] = (data_list[(data_dict['count']-1)//2] + data_list[data_dict['count']//2]) / 2
    return data_dict
    
def std(data_list,data_dict):
    if 'std' in data_dict.keys():
        return
    var(data_list,data_dict)
    try:
        data_dict['std'] = math.sqrt(data_dict['var'])
    except ValueError as err:
        print(data_list)
        print(data_dict['mean'])
        print(data_dict['var'])
        ValueError(err)
    return data_dict
    
def skewness(data_list,data_dict):
    if 'skew' in data_dict.keys():
        return
    std(data_list,data_dict)
    try:
        data_dict['skew'] = sum([(x-data_dict['mean'])**3 for x in data_list])/(data_dict['count']*data_dict['std']**3)
    except ZeroDivisionError:
        data_dict['skew'] = 'NaN'
    return data_dict
    
def kurtosis(data_list,data_dict):
    if 'kurt' in data_dict.keys():
        return
    std(data_list,data_dict)
    try:
        data_dict['kurt'] = sum([(x-data_dict['mean'])**4 for x in data_list])/(data_dict['count']*data_dict['std']**4) - 3
    except ZeroDivisionError:
        data_dict['kurt'] = 'NaN'
    return data_dict

def iqr(data_list,data_dict):
    if 'iqr' in data_dict.keys():
        return
    count(data_list, data_dict)
    data_list.sort()
    cnt = (int(data_dict['count']))//2
    l_half = data_list[:cnt]
    u_half = data_list[-cnt:]
    q1 = (l_half[(len(l_half)-1)//2] + l_half[len(l_half)//2]) / 2
    q3 = (u_half[(len(u_half)-1)//2] + u_half[len(u_half)//2]) / 2
    data_dict['iqr'] = q3-q1
    return data_dict

# Dictionary of input statistics
STATS = {'mean': mean,
         'median': median,
         'std': std,
         'var': var,
         'skew': skewness,
         'kurt': kurtosis,
         'count': count,
         'max': maxval,
         'min': minval,
         'iqr': iqr}

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Calculate simple statistics to groups of data in a csv file.')
    parser.add_argument('--statistics', dest='statistics', type=str,
                        help='Types of statistics to calculate', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input csv collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    if args.statistics == 'all':
        statistics = [i for i,v in STATS.items() ]
    else:
        statistics = args.statistics.split(',')
    logger.info('statistics = {}'.format(statistics))
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Get a list of all input files
    csv_files = [f for f in Path(inpDir).iterdir() if f.name.endswith('csv')]

    # Open each csv files
    for feat_file in csv_files:
        fpath = str(feat_file.absolute())
        out = str(Path(outDir).joinpath(feat_file.name).absolute())
        with open(fpath,'r') as fr:
            with open(str(out),'w') as fw:
                # Read the first line, which should contain headers
                first_line = fr.readline()
                headers = first_line.rstrip('\n').split(',')
                var_ind = {key:val for key,val in enumerate(headers)} # map headers to line positions
                # If no column is labeled file, throw an error
                if 'file' not in headers:
                    ValueError('At least one column must have a header title file.')

                # Generate the output dictionary template and format string
                line_dict = {'file': 'NaN'}
                for key in headers:
                    if key == 'file':
                        continue
                    for stat in statistics:
                        line_dict[key + '_' + stat] = 'NaN'

                        # Generate the line template
                line_template = ','.join([('{' + h + '}') for h in line_dict.keys()]) + '\n'

                # Write headers to the new file
                fw.write(','.join(line_dict.keys()) + '\n')

                # Get the first line of data
                line = fr.readline()

                # Read each line in the stitching vector
                fnum = 0
                fcheck = 0
                while line:
                    # Parse the current line as a dictionary
                    p_line = {var_ind[ind]:val for ind,val in enumerate(line.rstrip('\n').split(','))}
                    for key,val in p_line.items():
                        v = get_number(val)
                        p_line[key] = [v]

                    # Loop through rows until the filename changes
                    line = fr.readline()
                    np_line = {var_ind[ind]:val for ind,val in enumerate(line.rstrip('\n').split(','))}
                    while line and p_line['file'][0] == np_line['file']:
                        # Store the values in a feature list
                        for key,val in np_line.items():
                            if isinstance(val,str):
                                p_line[key].append(get_number(val[0]))

                        # Get the next line
                        line = fr.readline()
                        np_line = {var_ind[ind]:val for ind,val in enumerate(line.rstrip('\n').split(','))}

                    # Get the mean of the feature list, save in the file dictionary
                    for key,val in p_line.items():
                        # Set the file name
                        if key=='file':
                            line_dict['file'] = val[0]
                            continue

                        # Grab only float values
                        inp_data = [d for d in val if isinstance(d,float)]

                        # If inp_data contains no floats, skip it
                        if len(inp_data) == 0:
                            continue

                        # Calculate the statistics for the feature
                        data_dict = {}
                        for stat in statistics:
                            STATS[stat](inp_data,data_dict)
                            line_dict[key + '_' + stat] = data_dict[stat]
                    fw.write(line_template.format(**line_dict))
                    line_dict = {key:'NaN' for key in line_dict.keys()}

                    # Checkpoint
                    fnum += 1
                    if fnum > fcheck:
                        fcheck += 1
                        logger.info('Unique Files parsed: {}'.format(fnum))