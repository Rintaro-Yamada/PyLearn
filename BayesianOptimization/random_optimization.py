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

    savefig_pass="sin_random_seed_"

    random.seed(seed)
    # 初期点の生成
    train_index = random.sample(index_list, initial_num)

    random_index=random.sample(index_list, 100)
    X_train = X[train_index]
    y_train = y[train_index]
    
    result_dir_path = "./result/"
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path + savefig_pass + str(seed)])

    x_next = X[random_index[X_train.ndim]]

    #獲得関数の名前
    label_name="Random"
    
    # simple regretを計算してlistで各イテレーションの推移を記録
    true_max = y.max()
    simple_regret = true_max - y_train.max()
    simple_regret_list = [simple_regret]


    # ベイズ最適化のイテレーションを回す
    for i in range(max_iter):
        next_index = random_index[i]
        print(X_train)
        #next_index = np.argmax(acquisition_function)
        x_next = X[next_index]
        y_next = y[next_index]
        X_train = np.append(X_train, [x_next], axis=0)
        y_train = np.append(y_train, [y_next], axis=0)
        #simple regret を計算
        simple_regret = true_max - y_train.max()
        simple_regret_list.append(simple_regret)
        #観測データを更新

    np.savetxt(result_dir_path + savefig_pass + str(seed) + "/simple_regret.csv", np.array(simple_regret_list), delimiter=",")


def main():
    argv = sys.argv
    seed = 0
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    # 初期点を変えた10通りの実験を並列に行う (詳しくは公式のリファレンスを見てください)
    parallel_num = 10
    _ = Parallel(n_jobs=parallel_num)([
        delayed(experiment_each_seed)(i, initial_num, max_iter) for i in range(parallel_num)
    ])

if __name__=="__main__":
    main()