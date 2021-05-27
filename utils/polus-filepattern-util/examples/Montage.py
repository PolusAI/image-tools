import bfio, filepattern, pathlib, pprint

filepath = pathlib.Path(__file__).parent
filepath = filepath.joinpath('data/Small_Fluorescent_Test_Dataset/image-tiles/')

fp = filepattern.FilePattern(filepath,'img_r00{y}_c00{x}.tif')

pprint.pprint(fp.get_matching(X=1))

# for file in fp(group_by='y'):
    
#     pprint.pprint(file)
