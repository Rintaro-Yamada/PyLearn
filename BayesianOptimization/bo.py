import numpy as np
import subprocess
import sys
import random
import math
import csv
from nptyping import NDArray
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from scipy.stats import norm
from sklearn.utils import check_array, check_random_state, as_float_array
from joblib import Parallel, delayed
from ThetaGenerator import ThetaGenerator
from RBFKernel import RBFKernel

def func(x: NDArray[float]) -> NDArray[float]: #テスト用のforrester関数
    # 最大化したいので符号を反転
    return 0.1*(10*x-2)**2+5*np.sin(15*x)
    #return (-1) * (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def RFM(x:NDArray[float],dim:int,omega:NDArray[float],b:NDArray[float],variance:float) -> NDArray[float]:
    phi=np.sqrt(variance*2/dim)*(np.cos(omega.T*x+b.T))
    return phi

def plot(seed, mu: NDArray[float], var: NDArray[float],X:NDArray[float],y:NDArray[float],X_train:NDArray[float],y_train:NDArray[float],i:int):
    plt.rcParams["font.size"] = 13
    plt.subplot(1, 1, 1)
    plt.title("Gaussian Process")
    plt.plot(X.ravel(), y, "g--", label="true")
    plt.plot(X.ravel(), mu, "b", label="pred_mean")
    plt.fill_between(X.ravel(), (mu + 2 * np.sqrt(var)).ravel(), (mu - 2 * np.sqrt(var)).ravel(), alpha=0.3, color="blue")
    plt.plot(X_train.ravel(), y_train.ravel(), "ro", label="observed")
    plt.legend(loc="lower left",prop={'size': 8})
    plt.savefig(result_dir_path + savefig_pass +str(seed)+"/bayesian_" + str(i) +".pdf") 
    plt.close()

def MES(y_star:NDArray[float], pred_mu:NDArray[float], pred_var:NDArray[float], k:int, train_index:NDArray[int]) -> float:
    y_sample = np.tile(y_star,(pred_mu.shape[0],1))
    gamma_y = (y_sample.T-pred_mu)/np.sqrt(pred_var)
    psi_gamma = norm.pdf(gamma_y,loc=0,scale=1)
    large_psi_gamma = norm.cdf(gamma_y, loc=0, scale=1)
    #log_large_psi_gamma = norm.logcdf(gamma_y, loc=0, scale=1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_y*psi_gamma
    B = 2*large_psi_gamma
    temp=A/B-log_large_psi_gamma
    alpha=np.sum(temp,axis=0)/k
    #alpha[train_index]=0 #観測済みの点の獲得関数値は0にする

    return alpha

def expected_improvement(X_train: NDArray[float], y_train: NDArray[float], pred_mean: NDArray[float], pred_var: NDArray[float]) -> NDArray[float]:
    tau=y_train.max()
    tau=np.full(pred_mean.shape,tau)
    t=(pred_mean-tau)/np.sqrt(pred_var)
    #norm.cdf、norm.pdfはscipy.statsのライブラリ。それぞれ標準正規分布の累積密度関数と、密度関数を示す
    acq=(pred_mean-tau)*norm.cdf(x=t, loc=0, scale=1)+np.sqrt(pred_var)*norm.pdf(x=t, loc=0, scale=1)
    return acq

def upper_confidence_bound(X_train: NDArray[float],pred_mean: NDArray[float], pred_var: NDArray[float]) -> NDArray[float]:
    t = X_train.shape[0]
    #N = X_train.shape[0]
    #print(t)
    acq = pred_mean + np.sqrt(2*np.log(t ** 2 + 1)) * np.sqrt(pred_var)
    #acq = pred_mean + np.sqrt(np.log10(N)/N) * np.sqrt(pred_var)
    return acq

'''
    #分布のプロットをしてみる
    plt.title("distribution")
    ax1.plot(X.ravel(),y_sample.T.ravel(),"g",label="y_star")
    ax1.plot(X.ravel(),pred_mu,"b",label="pred_mu")
    ax1.plot(X.ravel(),((y_sample.T-pred_mu.ravel())).ravel(),"r",label="y_star-pred_mu")
    ax1.legend(loc="lower left",prop={'size': 8})
    ax2 = fig.add_subplot(2, 2, 2) 
    ax2.plot(X.ravel(),gamma_y.ravel(),"r",label="gamma_y")
    ax2.legend(loc="upper left",prop={'size': 8})
    ax3 = fig.add_subplot(2,2,3)
    #ax3.plot(X.ravel(),((y_sample.T-pred_mu.ravel())).ravel(),"r",label="pred_mu-pred_var")
    ax3.plot(X.ravel(),np.sqrt(pred_var),"g",label="sqrt(pred_var)")
    ax3.legend(loc="lower left",prop={'size': 8})
    plt.savefig(result_dir_path + savefig_pass +str(seed)+"/gamma_y"+".pdf")
'''

def experiment(seed: int, initial_num: int, max_iter: int):
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path + savefig_pass + str(seed)])
    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    grid_num = 500
    index_list = range(grid_num)
    X = np.c_[np.linspace(-1, 1, grid_num)]
    y = func(X)
    #regret=np.empty(0)
    
    #初期点の生成
    random.seed(seed)
    np.random.seed(seed)
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    #初期点のsimple regret
    train_regret_max = y_train.max(axis=0)
    true_regret_max = y.max(axis=0)
    regret = np.array(true_regret_max-train_regret_max)

    #カーネル行列の作成
    length_scale=0.1
    variance=5
    #観測誤差の固定
    noise_var = 1.0e-4
    kernel = RBFKernel(variance, length_scale)

    #key = True
    acq_name='MES'

    #max_iter回ベイズ最適化を行う
    for i in range (max_iter):
        K = kernel(X_train, X_train)
        # 精度(precision)行列の計算
        precision = np.linalg.inv(K + noise_var*np.eye(np.size(X_train)))
        #予測分布の導出
        k_star = kernel(X_train,X) #k*
        k_star_T_Kinv = np.dot(k_star.T, precision)
        pred_mu = np.dot(k_star_T_Kinv, y_train).ravel()
        pred_var = kernel(X, X) - np.dot(k_star_T_Kinv, k_star)
        pred_var_diag = np.diag(pred_var)

        #初期データの結果のプロット
        #plot(seed,pred_mu, pred_var_diag, X, y, X_train, y_train,i)

        if acq_name=='MES':
            dim=1000
            random_state = check_random_state(seed)
            omega = (np.sqrt(1/(length_scale**2)) * random_state.normal(
                size=(dim, 1)))
            #omega = np.c_[np.sqrt(2*0.2)*np.random.randn(dim)]

            b = np.c_[np.random.rand(dim) * 2 * np.pi] #[0,2π]の一様乱数

            #RFMから特徴量ベクトルΦ(x)を取得
            large_phi = RFM(X_train, dim,omega,b,variance) #D=100とした  10*1000

            #パラメータΘの分布の計算
            Theta = ThetaGenerator(dim,noise_var)
            Theta.calc(large_phi,y_train)

            #Xの特徴量ベクトル
            phi = RFM(X, dim,omega,b,variance)
            
            #パラメータΘの獲得. 引数は出したい関数の個数
            k=10
            theta=Theta.getTheta(k)
            #ブラックボックス関数fの近似を取得する。
            f_x = np.dot(theta,phi.T)

            #fの予測分布
            RFM_pred_mu=np.dot(phi,Theta.mu)
            temp=np.dot(phi,Theta.var)
            RFM_pred_var = np.dot(temp,phi.T)
            RFM_pred_var_diag = np.diag(RFM_pred_var)

            #MES
            y_star = f_x.max(axis=1)
            #y_star[y_star < y_train.max()+np.sqrt(noise_var)*5] = y_train.max()+np.sqrt(noise_var)*5

            #print(y_star)
            alpha = MES(y_star, pred_mu, pred_var_diag, k, train_index)
        
        if acq_name == 'UCB':
            alpha = upper_confidence_bound(X_train, pred_mu, pred_var_diag)

        if acq_name == 'EI':
            alpha = expected_improvement(X_train, y_train, pred_mu, pred_var_diag)

        next_index = np.argmax(alpha)

        #データの更新
        x_next = X[next_index] #候補点
        y_next = y[next_index]
        X_train = np.append(X_train, [x_next], axis=0)
        y_train = np.append(y_train, [y_next], axis=0)
        train_index.append(next_index)
        
        #simple_regret
        train_regret_max = y_train.max(axis=0)
        true_regret_max = y.max(axis=0)
        regret=np.append(regret,true_regret_max-train_regret_max)

        '''     
        #候補点とRFMによって得られた関数fの描写
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(3, 1, 1)
        plt.title("Observed=%2d, x_next=%f"%(initial_num+i, x_next))
        #fのプロット
           
        for j in range(k):
            ax1.plot(X.ravel(), f_x[j].ravel(), "b", label="f_"+str(j))
        ax1.plot(X.ravel(), y, "g--", label="true")
        ax1.plot(X.ravel(), RFM_pred_mu.ravel(), "r", label="RFM_pred_mean")
        ax1.fill_between(X.ravel(), (RFM_pred_mu.ravel() + 2 * np.sqrt(RFM_pred_var_diag)).ravel(), (RFM_pred_mu.ravel() - 2 * np.sqrt(RFM_pred_var_diag)).ravel(), alpha=0.3, color="green", label="RFM_pred_var")
        ax1.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
        ax1.legend(loc="lower left",prop={'size': 8})
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(X.ravel(), y, "g--", label="true")
        ax2.plot(X.ravel(), RFM_pred_mu.ravel(), "r", label="RFM_pred_mean")
        ax2.fill_between(X.ravel(), (RFM_pred_mu.ravel() + 2 * np.sqrt(RFM_pred_var_diag)).ravel(), (RFM_pred_mu.ravel() - 2 * np.sqrt(RFM_pred_var_diag)).ravel(), alpha=0.3, color="green", label="RFM_pred_var")
        ax2.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
        ax2.plot(X_train.ravel()[-1], y_next, "b*", label="x_next_point")
        ax2.legend(loc="lower left",prop={'size': 8})
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(X.ravel(),alpha,"g")
        plt.savefig(result_dir_path + savefig_pass +str(seed)+"/rfm_" + str(i) +".pdf")
        plt.close()
        '''

        ''' conflict 部分
        if acq_name == 'MES':
            #候補点とRFMによって得られた関数fの描写
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(3, 1, 1)
            plt.title("Observed=%2d, x_next=%f"%(initial_num+i, x_next))
            #fのプロット
            for j in range(k):
                ax1.plot(X.ravel(), f_x[j].ravel(), "b", label="f_"+str(j))
            ax1.plot(X.ravel(), y, "g--", label="true")
            ax1.plot(X.ravel(), RFM_pred_mu.ravel(), "r", label="RFM_pred_mean")
            ax1.fill_between(X.ravel(), (RFM_pred_mu.ravel() + 2 * np.sqrt(RFM_pred_var_diag)).ravel(), (RFM_pred_mu.ravel() - 2 * np.sqrt(RFM_pred_var_diag)).ravel(), alpha=0.3, color="green", label="RFM_pred_var")
            ax1.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
            ax1.legend(loc="lower left",prop={'size': 8})
            ax2 = fig.add_subplot(3, 1, 2)
            ax2.plot(X.ravel(), y, "g--", label="true")
            ax2.plot(X.ravel(), RFM_pred_mu.ravel(), "r", label="RFM_pred_mean")
            ax2.fill_between(X.ravel(), (RFM_pred_mu.ravel() + 2 * np.sqrt(RFM_pred_var_diag)).ravel(), (RFM_pred_mu.ravel() - 2 * np.sqrt(RFM_pred_var_diag)).ravel(), alpha=0.3, color="green", label="RFM_pred_var")
            ax2.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
            ax2.plot(X_train.ravel()[-1], y_next, "b*", label="x_next_point")
            ax2.legend(loc="lower left",prop={'size': 8})
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(X.ravel(),alpha,"g")
            plt.savefig(result_dir_path + savefig_pass +str(seed)+"/rfm_" + str(i) +".pdf")
            plt.close()
        '''
    #print(regret)
    plt.plot(range(max_iter+1), regret, "g", label="simple_regret")
    plt.legend()
    plt.savefig(result_dir_path + savefig_pass + str(seed) + "/simple_regret.pdf")
    plt.close()
    
    np.savetxt('result/seed' + str(seed) + '/regret_'+acq_name+'.csv', regret)

def main():
    argv = sys.argv
    initial_num = int(argv[1])
    max_iter = int(argv[2])


    #単体テスト用
    
    seed=2
    experiment(seed, initial_num, max_iter)
    '''
    # 初期点を変えた10通りの実験を並列に行う (詳しくは公式のリファレンスを見てください)
    parallel_num = 10
    _ = Parallel(n_jobs=parallel_num)([
        delayed(experiment)(l, initial_num, max_iter) for l in [i for i in range(0, 10)]
    ])
    '''
if __name__ == "__main__":
    savefig_pass="seed"
    result_dir_path = "./result/"
    main()


''' ガウス過程回帰モデル（MES）の作り方

1. 初期(訓練)データを用意 p81
2. 関数を用意
3. カーネル何使おうか？決める(今回はRBF)
4. カーネル行列の作成
5. RFMの作成
6. 関数fの近似
7. y*の獲得
8. 活性化関数の作成(MES)
9. 新しい入力点x'の出力y'を求める

'''
'''
for文内のomega betaのseedは変えるべきか？
'''