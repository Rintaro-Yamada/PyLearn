#fill_betweenの使い方
import numpy as np
import matplotlib.pyplot as plt
from math import *

x=np.arange(0,10,0.01)
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,color='k')
plt.plot(x,y2,color='k')

x1=np.arange(0,5,0.01)
z1=np.sin(x1)
z2=np.cos(x1)
plt.fill_between(x1,z1,z2,where=z1>z2,facecolor='y',alpha=0.5)
plt.show()