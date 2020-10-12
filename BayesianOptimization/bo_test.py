import numpy as np
import subprocess
import sys
import random
import math
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
    #return np.sin(5*x)
    return (-1) * (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def RFM(x:NDArray[float],dim:int,omega:NDArray[float],b:NDArray[float],variance:float) -> NDArray[float]:
    phi=np.sqrt(variance*2/dim)*(np.cos(omega.T*x+b.T))
    return phi

def plot(mu: NDArray[float], var: NDArray[float],X:NDArray[float],y:NDArray[float],X_train:NDArray[float],y_train:NDArray[float]):
    plt.rcParams["font.size"] = 13
    plt.subplot(1, 1, 1)
    plt.title("Gaussian Process")
    plt.plot(X.ravel(), y, "g--", label="true")
    plt.plot(X.ravel(), mu, "b", label="pred_mean")
    plt.fill_between(X.ravel(), (mu + 2 * np.sqrt(var)).ravel(), (mu - 2 * np.sqrt(var)).ravel(), alpha=0.3, color="blue")
    plt.plot(X_train.ravel(), y_train.ravel(), "ro", label="observed")
    plt.legend(loc="lower left",prop={'size': 8})
    plt.savefig(result_dir_path + savefig_pass +str(seed)+"/GP_initial.pdf")
    #plt.savefig("GP.pdf")
    plt.close()

def MES(f_x:NDArray[float], pred_mu:NDArray[float], pred_var:NDArray[float], k:int) -> float:
    y_star = f_x.max(axis=0)
    y_sample=np.tile(y_star,(pred_mu.shape[0],1))
    gamma_y=(y_sample.T-pred_mu)/np.sqrt(pred_var) #gamma_y D*K配2
    print(gamma_y)
    print(np.mean(gamma_y,axis=1))
    print(np.var(gamma_y,axis=1))
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(2, 2, 1)
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
    #plt.show()
    plt.close()
    #sys.exit()
    psi_gamma=norm.pdf(gamma_y,loc=pred_mu, scale=np.sqrt(pred_var))
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(2, 2, 1)
    plt.title("distribution")
    ax1.plot(X.ravel(),gamma_y.ravel(),"r",label="gamma_y")
    ax1.legend(loc="upper left",prop={'size': 8})
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(X.ravel(), psi_gamma.ravel(), "r", label="psi_gamma")
    ax2.legend(loc="upper left",prop={'size': 8})
    large_psi_gamma=norm.cdf(gamma_y, loc=pred_mu, scale=np.sqrt(pred_var))
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(X.ravel(), large_psi_gamma.ravel(), "r", label="large_psi_gamma")
    ax3.legend(loc="lower left",prop={'size': 8})
    ax4 = fig.add_subplot(2, 2, 4)
    log_large_psi_gamma=norm.logcdf(gamma_y, loc=pred_mu, scale=np.sqrt(pred_var))
    ax4.plot(X.ravel(), log_large_psi_gamma.ravel(), "r", label="log_large_psi_gamma")
    ax4.legend(loc="lower left",prop={'size': 8})
    temp=np.divide((gamma_y*psi_gamma),(2*large_psi_gamma),out=np.zeros_like(gamma_y*psi_gamma), where=(2*large_psi_gamma)!=0)-log_large_psi_gamma
    alpha=np.sum(temp,axis=0)/k
    plt.savefig(result_dir_path + savefig_pass +str(seed)+"/distribution_"+".pdf")
    plt.close()
    return alpha


def experiment(seed: int, initial_num: int, max_iter: int):
    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    
    
    #初期点の生成
    random.seed(seed)
    #np.random.seed(seed)
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    #カーネル行列の作成
    length_scale=0.1
    variance=5
    #観測誤差の固定
    noise_var = 1.0e-4
    kernel=RBFKernel(variance,length_scale)

    key=True

    #max_iter回ベイズ最適化を行う
    
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
    if key:
        plot(pred_mu, pred_var_diag, X, y, X_train, y_train)
        key=False

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
    theta=Theta.getTheta(1)
    #ブラックボックス関数fの近似を取得する。
    f_x = np.dot(theta,phi.T)

    #fの予測分布
    RFM_pred_mu=np.dot(phi,Theta.mu)
    temp=np.dot(phi,Theta.var)
    RFM_pred_var = np.dot(temp,phi.T)
    RFM_pred_var_diag = np.diag(RFM_pred_var)
    #y*の獲得
    alpha = MES(f_x, pred_mu, pred_var_diag, 1) #k=8
    next_index = np.argmax(alpha)

    x_next = X[next_index] #候補点
    y_next = y[next_index]
    X_train = np.append(X_train, [x_next], axis=0)
    y_train = np.append(y_train, [y_next], axis=0)

    #獲得関数の値をプロットしてみる
    plt.title("point")
    plt.plot(X.ravel(),alpha,"g")
    plt.savefig(result_dir_path + savefig_pass +str(seed)+savefig_pass_alpha+"/alpha_"+".pdf")
    plt.close()
    

    #候補点とRFMによって得られた関数fの描写
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.title("Observed=%2d, x_next=%f"%(initial_num, x_next))
    #fのプロット
   
    ax1.plot(X.ravel(), f_x.ravel(), "b", label="f_0")
    ax1.plot(X.ravel(), y, "g--", label="true")
    ax1.plot(X.ravel(), pred_mu.ravel(), "r", label="pred_mean")
    ax1.fill_between(X.ravel(), (RFM_pred_mu.ravel() + 2 * np.sqrt(RFM_pred_var_diag)).ravel(), (RFM_pred_mu.ravel() - 2 * np.sqrt(RFM_pred_var_diag)).ravel(), alpha=0.3, color="green", label="RFM_pred_var")
    ax1.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
    ax1.legend(loc="lower left",prop={'size': 8})
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(X.ravel(), y, "g--", label="true")
    ax2.plot(X.ravel(), RFM_pred_mu.ravel(), "r", label="RFM_pred_mean")
    ax2.fill_between(X.ravel(), (RFM_pred_mu.ravel() + 2 * np.sqrt(RFM_pred_var_diag)).ravel(), (RFM_pred_mu.ravel() - 2 * np.sqrt(RFM_pred_var_diag)).ravel(), alpha=0.3, color="green", label="RFM_pred_var")
    ax2.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
    ax2.plot(X_train.ravel()[-1], y_next, "b*", label="x_next_point")
    ax2.legend(loc="lower left",prop={'size': 8})
    plt.savefig(result_dir_path + savefig_pass +str(seed)+"/rfm_"+".pdf")
    plt.close()
    
def main():
    argv = sys.argv
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    experiment(seed,initial_num,max_iter)
    
    # 初期点を変えた10通りの実験を並列に行う (詳しくは公式のリファレンスを見てください)
    #parallel_num = 1
    #_ = Parallel(n_jobs=parallel_num)([
    #    delayed(experiment)(k, initial_num, max_iter) for k in range(parallel_num)
    #])
    
if __name__ == "__main__":
    grid_num = 200
    index_list = range(grid_num)
    X = np.c_[np.linspace(0, 1, grid_num)]
    y = func(X)
    seed = 5
    savefig_pass="RFM_seed"
    savefig_pass_alpha ="/alpha"
    result_dir_path = "./result/"
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path + savefig_pass + str(seed)+savefig_pass_alpha])
    main()


''' ガウス過程回帰モデル（MES）の作り方

1. 初期(訓練)データを用意 p81
2. 関数を用意
3. カーネル何使おうか？決める(今回はRBF)
4. カーネル行列の作成
5. RFMの作成
6. 関数fの近似
7. y*の獲得
5. 活性化関数の作成(MES)
5. 新しい入力点x'の出力y'を求める

'''
'''
for文内のomega betaのseedは変えるべきか？
'''