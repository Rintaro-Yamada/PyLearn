import numpy as np
import random
import sys
import subprocess
from joblib import Parallel, delayed
import pickle

from kernel_fn import RBFkernel
from blackbox_fn import forrester
from function_generator import FunctionGenerator
from aquisition_fn import ucb, ei, mes
from plot_lib import predict_plot, rfm_fn_plot
from hyper_opt import hyper_opt,opt

def experiment(seed, initial_num, max_iter, acq_name):

    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    grid_num = 200
    X = np.c_[np.linspace(0, 1, grid_num)]
    y = np.c_[forrester(X)]

    # 乱数シード値の固定(重要: 実験が再現できるようにするため)
    np.random.seed(seed)
    random.seed(seed)

    # 訓練データの作成
    index_list = range(grid_num)
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    # カーネルの設定
    '''
    :param lentgh_scale: RBFカーネルのバンド幅
    :param variance: 分散パラメータ
    :param noise_var: 観測誤差
    '''
    noise_var = 1.0e-4

    # プロットや結果保存のためのディレクトリをつくる (実験結果をいつでも復元できるようにいろんなログはとっておいて損はない)
    result_dir_path = "./"+acq_name+"/seed_"+str(seed)+"/"
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path])

    # ベイズ最適化
    for iter in range(max_iter):
        if iter % 5 == 0:  # 5イテレーションごとにハイパラ更新
            # ハイパラ最適化
            hyper_grid = 100
            length_scale, variance = opt(X_train, y_train, hyper_grid)
            kernel = RBFkernel(length_scale, variance, noise_var)
        pred_mean, pred_var = kernel.predict(kernel, X, X_train, y_train)
        # 結果のプロット
        #predict_plot(X, y, X_train, y_train, pred_mean, pred_var, result_dir_path + acq_name + "_iter" + str(iter) + ".pdf")
        
        if acq_name == 'ucb':
            alpha = ucb(X_train, pred_mean, pred_var)
        if acq_name == 'mes':
            F = FunctionGenerator(seed, length_scale, variance, noise_var, X)
            func_num = 5
            f = F.gen(X_train, y_train, func_num=func_num)
            # RFMからサンプリングされた関数のプロット
            # rfm_fn_plot(X, y, f, pred_mean, pred_var, "rfm_plot_" + str(iter) + ".pdf")
            y_star = f.max()
            print(y_star)
        
        next_index = np.argmax(alpha)

        #データの保存
        save_data_pass = result_dir_path+"iter_"+str(iter)+"/"
        _ = subprocess.check_call(["mkdir", "-p", save_data_pass])
        save_data_lists = [X, y, pred_mean, pred_var, alpha]
        data_name_lists = ["X", "y", "pred_mean", "pred_var", "alpha"]
        for i in range(len(data_name_lists)):
            with open(save_data_pass+data_name_lists[i]+".pickle",'wb') as f:
                pickle.dump(save_data_lists[i], f)
        

        # 訓練データの更新
        X_next = X[next_index]
        y_next = y[next_index]
        X_train = np.vstack((X_train, X_next))
        y_train = np.vstack((y_train, y_next))

def main():
    argv = sys.argv
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    acq_name = str(argv[3])

    '''
    便利そうなので, コマンドライン引数として, 以下を渡す.
    :param seed: シード値
    :param initial_num: 初期点数
    :param max_iter: ベイズ最適化のイテレーション数
    :param acq_name: 獲得関数名
    '''
    # 単体実験用
    seed = 1
    experiment(seed, initial_num, max_iter, acq_name)
    
    '''
    # 初期点を変えた10通りの実験を並列に行う (詳しくは公式のリファレンスを参照)
    parallel_num = 10
    _ = Parallel(n_jobs=parallel_num)([
        delayed(experiment)(i, initial_num, max_iter) for i in range(parallel_num)
    ])
    '''

if __name__ == "__main__":
    main()