import argparse
import boto3
import os
import time
import csv
import pathlib
import logging

""" Load the file extensions supported by bioformats """
with open(os.path.join('.','bflist.csv'),'r') as outfile:
    rdr = csv.reader(outfile)
    supported_formats = list(rdr)
    supported_formats = supported_formats[0]

""" Function that checks if a file extension is supported by bioformats """
def isBFormatsImage(fname):
    ext = pathlib.Path(fname).suffixes
    if len(ext)==0:
        return False
    if ''.join(ext)[1:] in supported_formats or ext[-1][1:] in supported_formats:
        return True
    else:
        return False

def main():
    """ Initialize the logger """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='s3import', description='Pull data from an S3 bucket.')

    """ Define the arguments """
    parser.add_argument('--s3Bucket',               # Name of the bucket
                        dest='s3_bucket',
                        type=str,
                        help='S3 bucket',
                        required=True)
    parser.add_argument('--s3Key',                  # Path to the data within the bucket
                        dest='s3_key',
                        type=str,
                        help='S3 bucket',
                        required=True)
    parser.add_argument('--awsAccessKey',           # AWS access key
                        dest='aws_access_key',
                        type=str,
                        help='The AWS access key id used to access the S3 bucket',
                        required=True)
    parser.add_argument('--awsAccessSecret',        # AWS secret key
                        dest='aws_access_secret',
                        type=str,
                        help='The AWS secret access key used to access the S3 bucket',
                        required=True)
    parser.add_argument('--outDir',                 # Output directory
                        dest='output_dir',
                        type=str,
                        help='The output directory for the data pulled from S3',
                        required=True)
    parser.add_argument('--getMeta',                # Get metadata instead of bioformats data
                        dest='get_metadata',
                        type=str,
                        help='If true, grabs metadata rather than images',
                        required=False)
    
    """ Get the input arguments """
    args = parser.parse_args()

    s3_bucket = args.s3_bucket
    s3_key = args.s3_key
    aws_access_key = args.aws_access_key
    aws_access_secret = args.aws_access_secret
    output_dir = args.output_dir
    get_metadata = args.get_metadata=='true'

    # If the key ends with a file separator, no files will be downloaded.
    # Remove trailing file separator if present.
    if s3_key[-1] == os.path.sep:
        s3_key = s3_key[0:-1]

    logger.info('s3_bucket = {}'.format(s3_bucket))
    logger.info('s3_key = {}'.format(s3_key))
    logger.info('aws_access_key = {}'.format(aws_access_key))
    logger.info('aws_access_secret = {}'.format(aws_access_secret))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('get_metadata = {}'.format(get_metadata))

    """ Initialize the S3 client """
    logger.info('Initializing boto3 client...')
    try:
        client = boto3.session.Session(aws_access_key_id=aws_access_key,
                                       aws_secret_access_key=aws_access_secret)    
        s3 = client.resource('s3')
        bucket = s3.Bucket(s3_bucket)
    except Exception:
        logger.exception("Failed to create an S3 session.")
    logger.info('Client initialized! Starting file transfer...')
    
    """ Download files """
    download_start = time.time()
    all_files = bucket.objects.all()
    for f in all_files:
        p, fname = os.path.split(f.key)
        
        # If the name is blank, it's a directory so skip
        if fname=="":
            continue
        # If the directory does not match the input, skip it
        if p!=s3_key:
            continue
        
        if get_metadata and isBFormatsImage(fname):
            logger.info("Skipping file: " + fname)
            continue
        if not isBFormatsImage(fname) and not get_metadata:
            logger.info("Skipping file: " + fname)
            continue
            
        if p==s3_key and fname != "":
            try:
                logger.info('{} >>> {}'.format(fname,output_dir + os.path.sep + fname))
                bucket.Object(f.key).download_file(output_dir + os.path.sep + fname)
            except Exception as e:
                logger.exception("Failed to download file: {}".format(fname))
    
    logger.info('All files downloaded in {} seconds!'.format(time.time() - download_start))
        
if __name__ == "__main__":
    main()