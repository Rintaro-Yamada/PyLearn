import matplotlib.pyplot as plt
import math
import numpy as np
from nptyping import NDArray

grid_num = 200
X=np.c_[np.linspace(0,1, grid_num)]
def func(X:NDArray[float]) -> NDArray[float]:
    return (-1) * (6 * X - 2) ** 2 * np.sin(12 * X - 4)
    return 

plt.plot(X,func(X),"b-",label=u'$f(x) = -(6x-2)^2\,\sin(12x-4)$')
plt.plot(X,func(X),"b-",label=u'$f(x) = -(6x-2)^2\,\sin(12x-4)$')
plt.savefig("forrester_func.pdf")
