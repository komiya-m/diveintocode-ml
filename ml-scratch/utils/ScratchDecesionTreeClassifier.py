import numpy as np
from sklearn.metrics import accuracy_score
import copy
import time
import matplotlib.pyplot as plt
% matplotlib
inline


class ScratchDecesionTreeClassifier():
    """
    決定木分類のスクラッチ実装

    Parameters
    ----------
   　max_depth : int
      最大のツリーの深さ

    Attributes
    ----------
    self.n : int
      Xのカラム数（特徴量の数）
    self.left : インスタンス
    　　左下のノード
    self.right : インスタンス
    　　右下のノード
    self.depth : int
      このインススタンスのツリーにおける深さ
    self.leaf : int
      リーフのクラス
    self.feature : int
      このノードの分割条件の特徴量のカラム
    self.threshold : float
      このノードの分割条件の閾値
    self.score : float
      このノードの分割条件の情報利得
    self.left : 次の形のndarray, shape (n_features,1)
      パラメータ
    """

    def __init__(self, max_depth=1):
        self.n = None
        self.left = None
        self.right = None
        self.max_depth = max_depth
        self.depth = None
        self.leaf = None
        self.feature = None
        self.threshold = None
        self.score = 0

    def fit(self, X, y):

        t0 = time.time()  # 時間の測定
        y = y.reshape(len(y), 1)  # リシェイプ
        Xy = np.hstack((X, y))  # Xy合成

        self._split_tree(Xy, depth=0)  # 学習の開始

        # 時間の表示
        t1 = time.time()
        print('time : {}s'.format(t1 - t0))

    def _split_tree(self, Xy, depth):
        # 再帰呼び出しでツリーを作成していく関数。

        self.depth = depth
        self.n = Xy.shape[1] - 1
        print("Recursion depth: " + str(self.depth))  # 現在のdepthを表示

        # 深さが設定まできたら、もしくはジニ不順度が０になったら分割を終了する
        if (self.depth == self.max_depth) or (self._gini_Impurity(Xy[:, -1]) == 0):
            self._leaf(Xy[:, -1])  # リーフの処理
            return
        # 分割条件の設定
        self._maximum_information_gain(Xy)
        print("Recursion score: {}".format(self.score))
        # ジニ不順度が下がらないようなら分割終了
        if self.threshold == None:
            self._leaf(Xy[:, -1])  # リーフの処理
            return
        # 条件の通りデータを分割する
        left_Xy = Xy[np.where(Xy[:, self.feature] >= self.threshold), :][0]  # np.whereでaxisが一つ増えているので[0]
        right_Xy = Xy[np.where(Xy[:, self.feature] < self.threshold), :][0]
        # 分割ノードのインスタンスを作成
        self.left = ScratchDecesionTreeClassifier(self.max_depth)
        self.right = ScratchDecesionTreeClassifier(self.max_depth)

        self.left._split_tree(left_Xy, depth + 1)  # 再帰呼び出し
        self.right._split_tree(right_Xy, depth + 1)  # 再帰呼び出し

    def predict(self, X):
        # 予測したnp.arrayを返す
        return np.array([self._predict_loop(X[i, :]) for i in range(len(X))])

    def _predict_loop(self, x):
        # 再帰呼び出しでツリーを降っていきクラスを予測する関数　
        # x: ndarray  shape(1, fetures)  １つだけのサンプルデータ

        # まずはリーフか判定 （xは１サンプルデータ）
        if not self.leaf == None:
            return self.leaf

        # 閾値以上ならノードを移動、リーフが見つかるまで繰り返す
        if x[self.feature] >= self.threshold:
            return self.left._predict_loop(x)  # 再帰呼び出し
        else:
            return self.right._predict_loop(x)  # 再帰呼び出し

    def _leaf(self, y):
        # leafに一番多いクラスを格納する

        label, count = np.unique(y, return_counts=True)
        self.leaf = label[np.argmax(count)]

    def _maximum_information_gain(self, Xy):
        # 情報利得を全特徴量と閾値で計算し、一番情報利得が高い分割条件をインスタンス変数へ

        y = Xy[:, -1]
        # 特徴量の数でループ
        for i in range(self.n):
            uni_X = np.unique(Xy[:, i])  # 閾値をユニークで取得
            uni_X = np.delete(uni_X, np.where(uni_X == uni_X.min()))  # 最小値はいらないので消す
            # 閾値の数でループ
            for threshold in uni_X:
                left_ind = np.where(Xy[:, i] >= threshold)[0]  # 閾値以上のindex抜き出す
                right_ind = np.where(Xy[:, i] < threshold)[0]  # 閾値以下のindex抜き出す
                IG = self._Information_gain(y, left_ind, right_ind)  # 情報利得の計算
                # 一番高い情報利得を残す
                if self.score < IG:
                    self.score = IG
                    self.threshold = threshold
                    self.feature = i

    def _gini_Impurity(self, y):
        # 入力されたyでジニ不純度を計算
        K = np.unique(y)  # クラスの数
        ntall = len(y)  # 総サンプル数
        nti = np.array([sum(y == i) for i in K])
        return 1 - sum((nti / ntall) ** 2)

    def _Information_gain(self, y, left_ind, right_ind):
        # 情報利得を計算
        npall = len(y)  # 親ノードのサンプル数
        nlall = len(left_ind)  # 左のサンプル数
        nrall = len(right_ind)  # 右のサンプル数
        return self._gini_Impurity(y) - nlall / npall * self._gini_Impurity(
            y[left_ind]) - nrall / npall * self._gini_Impurity(y[right_ind])

    def accuracy(self, y_test, y_pred):
        # accuracyを計算して返す
        return accuracy_score(y_test, y_pred)

    def plot_decision_area(self, X, y):
        # 決定領域をプロットします。

        X_label1 = X[y == 0, :]
        X_label2 = X[y == 1, :]

        plt.scatter(X_label1[:, 0], X_label1[:, 1], c="red", edgecolor="red", label="0")
        plt.scatter(X_label2[:, 0], X_label2[:, 1], c="blue", edgecolor="blue", label="1")

        # メッシュデータを作成
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                                       np.arange(x2_min, x2_max, 0.01))

        # メッシュデータ全部を学習モデルで分類
        z = self.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
        z = z.reshape(x1_mesh.shape)
        # メッシュデータと分離クラスを使って決定境界を描いている
        plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap="gnuplot")
        plt.xlim(x1_mesh.min(), x1_mesh.max())
        plt.ylim(x2_mesh.min(), x2_mesh.max())

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Determining region")
        plt.legend()

    def plot_decision_area_3class(self, X, y):
        # 決定領域をプロットします。3クラス版

        X_label1 = X[y == 0, :]
        X_label2 = X[y == 1, :]
        X_label3 = X[y == 2, :]

        plt.scatter(X_label1[:, 0], X_label1[:, 1], c="red", edgecolor="red", label="0")
        plt.scatter(X_label2[:, 0], X_label2[:, 1], c="blue", edgecolor="blue", label="1")
        plt.scatter(X_label3[:, 0], X_label3[:, 1], c="green", edgecolor="green", label="2")

        # メッシュデータを作成
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                                       np.arange(x2_min, x2_max, 0.01))

        # メッシュデータ全部を学習モデルで分類
        z = self.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
        z = z.reshape(x1_mesh.shape)
        # メッシュデータと分離クラスを使って決定境界を描いている
        plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, colors=['red', 'blue', 'green', 'pink', 'yellow'])
        plt.xlim(x1_mesh.min(), x1_mesh.max())
        plt.ylim(x2_mesh.min(), x2_mesh.max())

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Determining region")
        plt.legend()