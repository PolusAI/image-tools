import filepattern

img_dir = (
    "/Users/antoinegerardin/RT-CETSA-Analysis/.data/TEST_DASH2/1_extract_plates/images"
)
pattern = "{index:d+}_{temp:fffff}.ome.tiff"
fp = filepattern.FilePattern(img_dir, pattern)


f = fp()
