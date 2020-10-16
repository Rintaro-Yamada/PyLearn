import numpy as np
import matplotlib.pyplot as plt
import sys
def main():
    save_fig_name='./result/'+'regrets.pdf'
    plt.title('regrets')
    acq = ['UCB','EI']
    for acq_name in acq:
        regret=np.empty([0,10])
        for seed in range(5):
            file_pass = './result/seed'+str(seed)+'/'+'regret_'+acq_name+'.csv'
            regret = np.append(regret, [np.loadtxt(file_pass)], axis=0)
        
        mean=np.mean(regret,axis=0)
        unbiased_var = np.var(regret, axis=0, ddof=1)
        #print(unbiased_var)
        std_error = np.sqrt(unbiased_var / 10)
        print(std_error)
        upper_bound = mean + 2 * std_error
        lower_bound = mean - 2 * std_error
        if acq_name=='MES':
            plt.plot(range(mean.shape[0]),mean.ravel(),"b",label=acq_name)
            plt.fill_between(range(mean.shape[0]),upper_bound,lower_bound,alpha=0.1,color="blue")

        if acq_name=='UCB':
            plt.plot(range(mean.shape[0]),mean.ravel(),"r",label=acq_name)
            plt.fill_between(range(mean.shape[0]),upper_bound,lower_bound,alpha=0.1,color="red")

        if acq_name=='EI':
            plt.plot(range(mean.shape[0]),mean.ravel(),"g",label=acq_name)
            plt.fill_between(range(mean.shape[0]),upper_bound,lower_bound,alpha=0.1,color="green")
        
    plt.legend()
    plt.savefig(save_fig_name)
    plt.close()

if __name__ == "__main__":
    main()    