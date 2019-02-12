import numpy as np
from sklearn.metrics import accuracy_score
import copy
import matplotlib.pyplot as plt
% matplotlib
inline


class ScratchLogisticRegression():
    """
    ロジスティック回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    bias : bool
      バイアス項を入れる場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue
    lambda : float
      正則化パラメーター

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,1)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録
    self.n : int
      特徴量の数(バイアス含む)

    """

    def __init__(self, num_iter=500, lr=0.05, lam=0.02, bias=True, verbose=True):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.lam = lam
        self.bias = bias
        self.verbose = verbose
        self.n = 0
        self.coef_ = 0
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        # バイアスの処理
        if self.bias:
            m = len(X)
            X = np.hstack((np.ones(m).reshape(m, 1), X))

            # X_valにもバイアスの追加
            if type(X_val) == np.ndarray:
                m_val = len(X_val)
                X_val = np.hstack((np.ones(m_val).reshape(m_val, 1), X_val))

        # 特徴量の数取得
        self.n = X.shape[1]
        # シータの初期化
        self._init_theta()
        # 勾配降下
        self._gradient_descent(X, y, X_val, y_val)

    def _sigmoid(self, z):
        """
        Parameters
        ---------
        z : np.array

        Returns
        -------
        g:  np.array
            sigmoid関数にzを入力した値
        """

        g = 1 / (1 + np.e ** -z)
        return g

    def _init_theta(self):
        """
        self.coef_ : 次の形のndarray, shape (n_features,)
        パラメータ
        をランダムに初期化します。
        """
        self.coef_ = np.random.rand(self.n, 1)
        # self.coef_ = np.zeros(self.n).reshape(self.n, 1)

    def predict(self, X):
        """
        ロジスティクス回帰を使い推定する。
        roundで整数化し返す。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティクス回帰による推定結果
        """

        if self.bias:
            m = len(X)
            X = np.hstack((np.ones(m).reshape(m, 1), X))

        hx = self._sigmoid(np.dot(X, self.coef_))

        #閾値でTrue,Falseで変換して,intに変換
        pred = (hx >= 0.5).astype(int)

        return pred

    def predict_proba(self, X):
        """
        ロジスティクス回帰を使い推定する。
        確率を返す。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティクス回帰による推定結果
        """

        if self.bias:
            m = len(X)
            X = np.hstack((np.ones(m).reshape(m, 1), X))

        hx = self._sigmoid(np.dot(X, self.coef_))

        return hx

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
        hx:  次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果

        """

        hx = self._sigmoid(np.dot(X, self.coef_))

        return hx

    def _compute_cost(self, hx, y):
        """
        コスト関数を計算して返す

        Parameters
        ----------
        hx : 次の形のndarray, shape (n_samples, １)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
         J : 次の形のndarray, shape (1,)
          平均二乗誤差
        """
        # hxはシグモイドにX*theta.T入れたもの
        # hx = self._linear_hypothesis(X)
        m = len(y)
        y = y.reshape(m, 1)  # リシェイプ
        theta = copy.deepcopy(self.coef_)

        if self.bias:
            theta[0] = 0  # バイアス項は正則化しないため、ゼロを入れる

        J = 1 / m * sum(sum(-y * np.log(hx) - (1 - y) * np.log(1 - hx))) + self.lam / (2 * m) * sum(theta ** 2)

        return J

    def _gradient_descent(self, X, y, X_val=None, y_val=None):
        """
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        alpha = self.lr
        m = len(X)
        # y = y[:, np.newaxis]  # newaxisで縦ベクトルに変換
        y = y.reshape(m, 1)  # リシェイプ
        hx = self._linear_hypothesis(X)  # シグモイド済み

        for i in range(self.iter):
            theta = copy.deepcopy(self.coef_)
            if self.bias:
                theta[0] = 0  # バイアス項は正則化しないため、ゼロを入れる

            self.coef_ = self.coef_ - (alpha * (1 / m) * np.dot(X.T, (hx - y)) + self.lam / m * theta)
            hx = self._linear_hypothesis(X)

            # trainデータのlossのリザルトを出す
            if self.verbose:
                self.loss[i] = self._compute_cost(hx, y)

            # X_val入力ある場合MSEのリザルトをだす
            if type(X_val) == np.ndarray:
                val_pred = self._linear_hypothesis(X_val)
                self.val_loss[i] = self._compute_cost(val_pred, y_val)

    def plot_loss_train_and_val(self):
        """
        学習曲線をプロットします。

        loss : array
        一回ごとの勾配降下方のロスのログ(train)
         val_los : array
        一回ごとの勾配降下方のロスのログ(val or test)
        """
        plt.figure(figsize=(8, 5))
        plt.title("model_loss")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.plot(self.loss, label="train_loss")
        plt.plot(self.val_loss, label="val_loss")
        plt.yscale("log")
        plt.legend()

    def accuracy(self, y_test, y_pred):
        # accuracyを計算して返す
        return accuracy_score(y_test, y_pred)

    ##下の関数はメモです
    def validator(theta):
        # (x,y)におけるy軸。
        # シグモイド関数では θTxa=0 が分類AとBの境界だったので、その時の値を求めてプロットする。
        # θTx = θ0＊x0 + θ1＊x1 + θ2＊x2 = 0
        # x2  = -(θ0 + θ1＊x1) / θ2
        xline = np.linspace(-4, 4, 100)  # X１の線
        budary = -(theta[1] + theta[2] * xline) / theta[2]

        return budary