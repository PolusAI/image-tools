import re
import os
import csv

"""
This code takes a text file containing bioformats supported file
extensions and exports them as a csv. This code should be run
when upgrading to a new version of bioformats.
"""

supported_formats = []

with open('.' + os.path.sep + 'bflist_6-0-1.txt') as f:
    for line in f:
        formats = re.search(r'\(([\w\.,\s]+)+\)',line)
        if formats:
            supported_formats.extend(str.split(formats.group(1),', '))
    
remove_from_list = ['txt','csv']
remove_index = []
supported_formats.sort()

for f in range(0,len(supported_formats)):
    print('{}: {}'.format(f,supported_formats[f]))
    if supported_formats[f] in remove_from_list:
        remove_index.append(f)
        continue
    if f>0 and supported_formats[f] == supported_formats[f-1]:
        remove_index.append(f)

for f in range(len(remove_index),0,-1):
    del supported_formats[remove_index[f-1]]
    
with open(os.path.join('.','src','bflist.csv'),'w') as outfile:
    wr = csv.writer(outfile)
    wr.writerow(supported_formats)