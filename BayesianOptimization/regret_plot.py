import numpy as np
import matplotlib.pyplot as plt
import sys
def main():
    save_fig_name='./result/'+'regrets.pdf'
    plt.title('regrets')
    acq = ['MES','UCB','EI']
    for acq_name in acq:
        regret=np.empty([0,10])
        for seed in range(10):
            file_pass = './result/seed'+str(seed)+'/'+'regret_'+acq_name+'.csv'
            regret=np.append(regret,[np.loadtxt(file_pass)],axis=0)
        #print(regret)
        mean=np.mean(regret,axis=0)
        #print(mean.shape)
        std = np.std(regret,axis=0)

        if acq_name=='MES':
            plt.plot(range(mean.shape[0]),mean.ravel(),"b",label=acq_name)
            lower_bound =[mean-2*std if i>0 else 0 for i in mean-2*std]
            plt.fill_between(range(mean.shape[0]),mean+2*std,lower_bound,alpha=0.1,color="blue")

        if acq_name=='UCB':
            plt.plot(range(mean.shape[0]),mean.ravel(),"r",label=acq_name)
            plt.fill_between(range(mean.shape[0]),mean+2*std,lower_bound,alpha=0.1,color="red")

        if acq_name=='EI':
            plt.plot(range(mean.shape[0]),mean.ravel(),"g",label=acq_name)
            plt.fill_between(range(mean.shape[0]),mean+2*std,lower_bound,alpha=0.1,color="green")
        
    plt.legend()
    plt.savefig(save_fig_name)
    plt.close()

if __name__ == "__main__":
    main()    