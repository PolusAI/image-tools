import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df  = pd.read_csv("ProjectedData_EmbeddedSpace.csv")

x=df['Dimension1']
y=df['Dimension2']
#labels=df['Data']

labels  = pd.read_csv("mnist_784-Labels.csv")
labels2=np.int32(labels)
labels3=tuple(labels2.reshape(1,-1)[0])


scatter= plt.scatter(x,y, c = labels3, cmap = matplotlib.colors.ListedColormap(['red','green','blue','purple', 'yellow', 'black','magenta','brown','grey', 'cyan']), s=1)


classes=['0','1','2','3', '4', '5','6','7','8', '9']
plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc=7, bbox_to_anchor=(1.13, 0.5))

plt.savefig('Image.png')



#colors = ['red','green','blue','purple', 'yellow', 'blank','magenta','brown','grey', 'cyan']
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#labels  = pd.read_csv("mnist_784_Labels.csv")
#labels2 = np.float32(labels)

#plt.scatter(x,y,c=labels, cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(xx,yy,c=labels, cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(x,y, edgecolors='r')
#plt.xlim(-10, 10)
#plt.ylim(-10, 10)
#plt.savefig('myImage.png')

#plt.show()

#xx=np.float32(x)
#yy=np.float32(y)
