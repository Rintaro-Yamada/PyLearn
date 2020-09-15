import sys
import subprocess
import random
import numpy as np
import GPy
import math
from nptyping import NDArray
from scipy.stats import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def forrester(x: NDArray[float]) -> NDArray[float]:
    # 最大化したいので符号を反転
    #return (-1) * (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return 100*np.sin(x)/x

def expected_improvement(X: NDArray[float], pred_mean: NDArray[float], pred_var: NDArray[float], X_train: NDArray[float],xi: float = 0.0) -> NDArray[float]:
    #ガウス過程と機械学習本（p183の式(6.7)を参考)
    tau=forrester(X_train).max()
    tau=np.full(X.shape,tau)
    t=(pred_mean-tau)/pred_var
    
    #norm.cdf、norm.pdfはscipy.statsのライブラリ。それぞれ標準正規分布の累積密度関数と、密度関数を示す
    acq=(pred_mean-tau)*norm.cdf(x=t, loc=0, scale=1)+pred_var*norm.pdf(x=t, loc=0, scale=1)
    return acq
    #return np.zeros(X.shape)

def upper_confidence_bound(X: NDArray[float], pred_mean: NDArray[float],pred_var: NDArray[float],X_train: NDArray[float])-> NDArray[float]:
    t=X_train.ndim-1
    acq=pred_mean+math.sqrt(math.log(t^2)+1)*pred_var
    return acq

def experiment_each_seed(seed: int, initial_num: int, max_iter: int):
    '''
    初期点を生成する際のシードを引数のseedに固定したもとでベイズ最適化の実験を行う. 
    便利そうなので初期点の数はコマンドライン引数でしていできるようにしてみる. 

    :param seed: This is the seed value for generating initial data. 
    :param initial_num: This is the number of initial data. 
    '''

    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    grid_num = 200 
    index_list = range(grid_num)
    #X = np.c_[np.linspace(0, 1, grid_num)]
    X = np.c_[np.linspace(3, 20, grid_num)]
    y = forrester(X)

    savefig_pass="sin_ucb_seed_"

    random.seed(seed)
    # 初期点の生成
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    # GP model

    #sin関数はlengthscale=1.0
    #forresterはlengthsclae=0.25
    '''ハイパラめも

    0.2 のときは0.5くらいになった
    ucb_seed_2がむずい
    0.25で2はクリア。しかしucb_seed_0が失敗

    0.22で大成功
    '''

    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1,lengthscale=1.0)
    model = GPy.models.GPRegression(X_train, y_train, kernel=kernel, normalizer=True)
    #観測誤差の分散は適当に固定
    noise_var = 1.0e-4
    model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
    #model.optimize_restarts()
    print(model)
    pred_mean, pred_var = model.predict(X)
    # プロットや結果保存のためのディレクトリをつくる (実験結果をいつでも復元できるようにいろんなログはとっておいて損はない)
    result_dir_path = "./result/"
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path + savefig_pass + str(seed)])

    #x_next描写用に
    acquisition_function = upper_confidence_bound(X, pred_mean, pred_var,X_train)
    next_index = np.argmax(acquisition_function)
    x_next = X[next_index]

    #獲得関数の名前
    label_name="UCB"
    
    #回帰の様子をプロットしてみる (これは関数に切り出したほうがよさそうだが面倒なので今回はベタ書き)
    plt.rcParams["font.size"] = 13
    plt.subplot(2,1,1)
    plt.title("Observed=%2d, x_next=%f"%(initial_num, x_next))
    plt.plot(X.ravel(), y, "g--", label="true")
    plt.plot(X.ravel(), pred_mean, "b", label="pred mean")
    plt.fill_between(X.ravel(), (pred_mean + 2 * np.sqrt(pred_var)).ravel(), (pred_mean - 2 * np.sqrt(pred_var)).ravel(), alpha=0.3, color="blue")
    plt.plot(X_train.ravel()[:len(X_train)], y_train[:len(X_train)], "ro", label="observed")
    plt.legend(loc="lower left",prop={'size': 8})
    #plt.ylim([-40,20])
    plt.subplot(2,1,2)
    plt.plot(X.ravel(), upper_confidence_bound(X, pred_mean, pred_var,X_train), "g-", label=label_name)
    plt.legend(loc="lower left",prop={'size': 8})
    #plt.tight_layout()
    plt.savefig(result_dir_path + savefig_pass +str(seed)+"/predict_initial.pdf")
    plt.close()
    # simple regretを計算してlistで各イテレーションの推移を記録
    true_max = y.max()
    simple_regret = true_max - y_train.max()
    simple_regret_list = [simple_regret]

    

    # ベイズ最適化のイテレーションを回す
    for i in range(max_iter):
        acquisition_function = upper_confidence_bound(X, pred_mean, pred_var,X_train)
        next_index = np.argmax(acquisition_function)
        x_next = X[next_index]
        y_next = y[next_index]
        X_train = np.append(X_train, [x_next], axis=0)
        y_train = np.append(y_train, [y_next], axis=0)
        #simple regret を計算
        simple_regret = true_max - y_train.max()
        simple_regret_list.append(simple_regret)
        #観測データを更新
        model.set_XY(X_train, y_train)
        #model.optimize_restarts()
        pred_mean, pred_var = model.predict(X)

        #x_next描写用に
        acquisition_function = upper_confidence_bound(X, pred_mean, pred_var,X_train)
        next_index = np.argmax(acquisition_function)
        x_next = X[next_index]

        plt.rcParams["font.size"] = 13
        plt.subplot(2,1,1)
        plt.title("Observed=%2d, x_next=%f"%(initial_num+i+1, x_next))
        #plt.ylim(, Yの最大値)
        plt.plot(X.ravel(), y, "g--", label="true")
        plt.plot(X.ravel(), pred_mean, "b", label="pred mean")
        plt.fill_between(X.ravel(), (pred_mean + 2 * np.sqrt(pred_var)).ravel(), (pred_mean - 2 * np.sqrt(pred_var)).ravel(), alpha=0.3, color="blue")
        plt.plot(X_train.ravel()[:len(X_train)-1], y_train[:len(X_train)-1], "ro", label="observed")
        plt.plot(X_train.ravel()[-1], y_next, "c*", label="observation")
        #plt.ylim([-40,20])
        plt.legend(loc="lower left",prop={'size': 8})

        plt.subplot(2,1,2)
        plt.plot(X.ravel(), upper_confidence_bound(X, pred_mean, pred_var,X_train), "g-", label=label_name)
        plt.legend(loc="lower left",prop={'size': 8})
        #plt.tight_layout()
        plt.savefig(result_dir_path + savefig_pass +str(seed)+"/predict_" + str(i) +".pdf")
        plt.close()
    np.savetxt(result_dir_path + savefig_pass + str(seed) + "/simple_regret.csv", np.array(simple_regret_list), delimiter=",")


def main():
    argv = sys.argv
    seed = 0
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    # 初期点を変えた10通りの実験を並列に行う (詳しくは公式のリファレンスを見てください)
    parallel_num = 1
    _ = Parallel(n_jobs=parallel_num)([
        delayed(experiment_each_seed)(i, initial_num, max_iter) for i in range(parallel_num)
    ])

if __name__=="__main__":
    main()