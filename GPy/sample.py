import GPy
import numpy as np
import matplotlib.pyplot as plt  

kernel=GPy.kern.RBF(input_dim=1,variance=1,lengthscale=0.2)

np.random.seed(0)
N_sim=100
x_sim=np.linspace(-1,1,N_sim)
x_sim=x_sim[:,None]
print(x_sim)
mu=np.zeros(N_sim)
cov=kernel.K(x_sim,x_sim)
y_sim=np.random.multivariate_normal(mu,cov,size=20)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for i in range(20):
    ax.plot(x_sim[:],y_sim[i,:])
fig.savefig("fig1.png")
