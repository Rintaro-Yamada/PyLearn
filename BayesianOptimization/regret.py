import sys
import matplotlib.pyplot as plt
import numpy as np


#イテレーション数
max_iter=50

#max_iterに初期点のregret分の1を足す
ei_regret_list=np.zeros(max_iter+1)
ucb_regret_list=np.zeros(max_iter+1)
random_regret_list=np.zeros(max_iter+1)

seed=10
result_dir_path="./result/"
for i in range(seed):
    #ファイルまでのパス
    ei_seed_dir_path=result_dir_path+"sin_seed_"+str(i)+"/simple_regret.csv"
    ucb_seed_dir_path=result_dir_path+"sin_ucb_seed_"+str(i)+"/simple_regret.csv"
    random_seed_dir_path=result_dir_path+"sin_random_seed_"+str(i)+"/simple_regret.csv"

    #ei_regretの読み込み
    f = open(ei_seed_dir_path, 'r')
    line = f.readline()
    #cntはsimple_regret.csvの行数カウンタ
    cnt=0
    while line:
        ei_regret_list[cnt]+=float(line.strip())
        cnt=cnt+1
        line = f.readline()
    f.close()

    #ucb_regretの読み込み
    f = open(ucb_seed_dir_path, 'r')
    line = f.readline()
    #cntはsimple_regret.csvの行数カウンタ
    cnt=0
    while line:
        ucb_regret_list[cnt]+=float(line.strip())
        cnt=cnt+1
        line = f.readline()
    f.close()

    #random_regretの読み込み
    f = open(random_seed_dir_path, 'r')
    line = f.readline()
    #cntはsimple_regret.csvの行数カウンタ
    cnt=0
    while line:
        random_regret_list[cnt]+=float(line.strip())
        cnt=cnt+1
        line = f.readline()
    f.close()

#retretの平均値
ei_regret_list=ei_regret_list/seed
ucb_regret_list=ucb_regret_list/seed
random_regret_list=random_regret_list/seed
plt.rcParams["font.size"] = 18
plt.plot(np.arange(51), ei_regret_list, "b-", label="ei_regret")
plt.plot(np.arange(51), ucb_regret_list, "r-", label="ucb_regret")
plt.plot(np.arange(51), random_regret_list, "g-", label="random_regret")
plt.xlabel('iteration')
plt.ylabel('regret')
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("sin_regret.pdf")


