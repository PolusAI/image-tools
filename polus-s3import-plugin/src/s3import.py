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
    ext = ''.join(pathlib.Path(fname).suffixes)[1:]
    if ext in supported_formats:
        return True
    else:
        return False

def main():
    """ Initialize the logger """
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    """ Initialize argument parser """
    logging.info("Parsing arguments...")
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
                        required=True)
    
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
    if s3_key[-1] == os.path.sep
        s3_key = s3_key[0:-1]

    logging.info('s3_bucket = {}'.format(s3_bucket))
    logging.info('s3_key = {}'.format(s3_key))
    logging.info('aws_access_key = {}'.format(aws_access_key))
    logging.info('aws_access_secret = {}'.format(aws_access_secret))
    logging.info('output_dir = {}'.format(output_dir))
    logging.info('get_metadata = {}'.format(get_metadata))

    """ Initialize the S3 client """
    logging.info('Initializing boto3 client...')
    try:
        client = boto3.session.Session(aws_access_key_id=aws_access_key,
                                       aws_secret_access_key=aws_access_secret)    
        s3 = client.resource('s3')
        bucket = s3.Bucket(s3_bucket)
    except Exception as e:
        logging.exception("Failed to create an S3 session.")
    logging.info('Client initialized! Starting file transfer...')
    
    """ Download files """
    download_start = time.time()
    all_files = bucket.objects.all()
    for f in all_files:
        p, fname = os.path.split(f.key)
        out_path = output_dir
        
        if get_metadata and isBFormatsImage(fname):
            logging.info("Skipping file: " + fname)
            continue
        if not isBFormatsImage(fname) and not get_metadata:
            logging.info("Skipping file: " + fname)
            continue
            
        if p==s3_key and fname != "":
            try:
                logging.info('{} >>> {}'.format(fname,out_path + os.path.sep + fname))
                bucket.Object(f.key).download_file(out_path + os.path.sep + fname)
            except Exception as e:
                logging.exception("Failed to download file: {}".format(fname))
    
    logging.info('All files downloaded in {} seconds!'.format(time.time() - download_start))
        
if __name__ == "__main__":
    main()