'''

Include simple testing here for the 3 plugins that are functional - can change values depending on arg type

'''

import sys
import os

#ApplyManualThreshold
for thresh in [1000, 100, 10]:
        os.system('python ../polus-imagej-threshold-apply-plugin/src/main.py --in1 ./INP1  --threshold {} --out ./OUTP1 --opName "ApplyManualThreshold"'.format(thresh))

#ApplyConstantThreshold
for thresh in [1000, 100, 10]:
        os.system('python ../polus-imagej-threshold-apply-plugin/src/main.py --in1 ./INP1  --threshold {} --out ./OUTP1 --opName "ApplyConstantThreshold"'.format(thresh))

##GaussRAISingleSigma
for sigma in [1000, 100, 10]:
        os.system('python ../polus-imagej-blurring-plugin/src/main.py --in1 ./INP1  --sigma {} --out ./OUTP1 --opName "GaussRAISingleSigma"'.format(thresh))
