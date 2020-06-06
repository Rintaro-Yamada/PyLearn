import numpy as np
import random
from matplotlib import pyplot as plt
import sys

# ガウスカーネルを関数化
# ぱっとみて何かわからん変数名(p q r)は出来得る限り避けましょう.
# 一般的にはガウスカーネルは観測誤差を含まないと思います.
# 少なくとも実装上は含まないようにしたほうが便利です.
# クラスとして定義するのが一番使い勝手が良いと思います.
class GaussianKernel():
    def __init__(self, sigma2_f, length_scale):
        self.sigma2_f = sigma2_f # 関数のスケールを調節する超パラ (pに対応します)
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        temp1 = np.c_[np.sum(X1**2, axis=1)]
        temp2 = np.c_[np.sum(X2**2, axis=1)]
        norm = temp1 + temp2.T - 2*np.dot(X1, X2.T)
        return self.sigma2_f * np.exp(- norm / (2*self.length_scale**2))


# def kernel(x, x_prime, p, q, r):
#     if x == x_prime:
#         delta = 1
#     else:
#         delta = 0

#     return p*np.exp(-1 * (x - x_prime)**2 / q) + ( r * delta)
'''
こういう数値実験でランダムな要素がある場合には, 実験の再現のために乱数の種の固定を行います
'''
random.seed(0)
np.random.seed(0)

# 元データの作成
n=100
num_observation = 20
data_x = np.linspace(0, 4*np.pi, n)
data_y = 2*np.sin(data_x) + 3*np.cos(2*data_x) + 5*np.sin(2/3*data_x) + np.random.randn(len(data_x))

# # 信号を欠損させて部分的なサンプル点を得る
# missing_value_rate = 0.2
# sample_index = np.sort(np.random.choice(np.arange(n), int(n*missing_value_rate), replace=False))
'''
choiceは復元抽出なので, 非復元抽出のsampleを使うべきですかね. randomのimportが必要ですが.
numpyのみでも書けますが, 少し面倒になる感じです.
あと, sortする必要とくに無いと思います.
加えて, missing_value_rateだと20%のデータが欠落しているように思えますが, 逆で20%のデータが使えるんですよね...
'''
sample_index = random.sample(range(n), num_observation)


fig2=plt.figure(figsize=(12, 5))
ax2 = fig2.add_subplot(111)
ax2.set_position([0.1,0.1,0.5,0.8])
plt.title('signal data', fontsize=20)

# 元の信号
plt.plot(data_x, data_y, 'x', color='green', label='correct signal')

# 部分的なサンプル点
plt.plot(data_x[sample_index], data_y[sample_index], 'o', color='red', label='sample dots')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
plt.savefig("signal.png")
plt.show()
plt.close()
'''
このぐらいのコードなら問題になりませんが, plotは使わなくなったらplt.close()で閉じないとエラー吐かれることが多いです.
'''

'''
これは好みもありますが, 自分は基本縦ベクトルにして取り扱います (np.c_[]で縦になる)
'''
# データの定義
xtrain = np.c_[np.copy(data_x[sample_index])]
print(xtrain)

ytrain = np.c_[np.copy(data_y[sample_index])]

xtest = np.c_[np.copy(data_x)]

# 各パラメータ値
sigma2_f = 1.0
length_scale = np.sqrt(0.2) #不自然なパラメータに見えますが, 元の山田くんのコードに合わせました.
sigma2_noise = 0.1
'''
for文は全て必要ないです
全部行列演算に直しましょう. 速度も可読性も良くなるはずです.
'''

kernel = GaussianKernel(sigma2_f=sigma2_f, length_scale=length_scale)
K = kernel(xtrain, xtrain)
# 精度(precision)行列の計算
precision = np.linalg.inv(K + sigma2_noise*np.eye(np.size(xtrain)))
#元のコードでは2回呼び出されていましたが, 逆行列の計算はO(N^3)で非常に重たいです. 可能なら変数として保存しましょう.

k_test_train = kernel(xtest, xtrain)
kK_ = k_test_train.dot(precision)

mu = np.dot(kK_, ytrain).ravel()
var = kernel(xtest, xtest) - np.dot(kK_, k_test_train.T)
#あえて共分散も全部計算しています. カーネルを対角行列のみの計算を可能にして, np.einsumを使えば最初から対角のみ計算できます.
var = np.diag(var).ravel()


# # 以下, ガウス過程回帰の計算の基本アルゴリズム
# train_length = len(xtrain)
# # トレーニングデータ同士のカーネル行列の下地を準備
# K = np.zeros((train_length, train_length))

# for x in range(train_length):
#     for x_prime in range(train_length):
#         K[x, x_prime] = kernel(xtrain[x], xtrain[x_prime], Theta_1, Theta_2, Theta_3)

# # 内積はドットで計算
# yy = np.dot(np.linalg.inv(K), ytrain)

# test_length = len(xtest)
# for x_test in range(test_length):

#     # テストデータとトレーニングデータ間のカーネル行列の下地を準備
#     k = np.zeros((train_length,))
#     for x in range(train_length):
#         k[x] = kernel(xtrain[x], xtest[x_test], Theta_1, Theta_2, Theta_3)

#     s = kernel(xtest[x_test], xtest[x_test], Theta_1, Theta_2, Theta_3)
'''
ここ予測分散sに観測誤差が乗っているのは少し変ですね.
加えて, xtestという配列にx_testというインデックスを与えるのは実装上ちょっと見づらいですかね.
'''

#     # 内積はドットで計算して, 平均値の配列に追加
#     mu.append(np.dot(k, yy))
#     # 先に『k * K^-1』の部分を(内積なのでドットで)計算
#     kK_ = np.dot(k, np.linalg.inv(K)) #2回目
#     # 後半部分との内積をドットで計算して, 分散の配列に追加
#     var.append(s - np.dot(kK_, k.T))

fig=plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.set_position([0.1,0.1,0.5,0.8])
plt.title('signal prediction by Gaussian process', fontsize=20)

# 元の信号
plt.plot(data_x, data_y, 'x', color='green', label='correct signal')
# 部分的なサンプル点
plt.plot(data_x[sample_index], data_y[sample_index], 'o', color='red', label='sample dots')

# 分散を標準偏差に変換
std = np.sqrt(var)

# ガウス過程で求めた平均値を信号化
plt.plot(xtest, mu, color='blue', label='mean by Gaussian process')
# ガウス過程で求めた標準偏差を範囲化 *範囲に関してはコード末を参照
plt.fill_between(xtest.ravel(), mu + 2*std, mu - 2*std, alpha=.2, color='blue', label= 'standard deviation by Gaussian process')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
plt.savefig("gaussian.png")
plt.show()
plt.close()

# 平均値±(標準偏差×2) … 95.4%の確率で範囲内に指定の数値が現れる