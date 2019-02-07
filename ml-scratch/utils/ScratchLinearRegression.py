import numpy as np


class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

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
    n : int
        特徴量の数(バイアス含む)

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,1)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter=500, lr=0.01, bias=True, verbose=True):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.n = 0
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
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        if self.bias:
            m = len(X)
            X = np.hstack((np.ones(m).reshape(m, 1), X))

        hx = np.dot(X, self.coef_)

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

        hx = np.dot(X, self.coef_)

        return hx

    def MSE(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        mse : numpy.float
          平均二乗誤差
        """

        m = len(y)
        y = y.reshape(m, 1)  # リシェイプ
        mse = (1 / (2 * m)) * sum((y_pred - y) ** 2)

        return mse

    def _compute_cost(self, X, y):
        """
        平均二乗誤差を計算する。MSEは共通の関数を作っておき呼び出す

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
         J : 次の形のndarray, shape (1,)
          平均二乗誤差
        """

        J = self.MSE(self._linear_hypothesis(X), y)

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
        y = y[:, np.newaxis]  # newaxisで縦ベクトルに変換
        hx = self._linear_hypothesis(X)

        for i in range(self.iter):
            self.coef_ = self.coef_ - alpha * (1 / m) * np.dot(X.T, (hx - y))
            hx = self._linear_hypothesis(X)

            # trainデータのlossのリザルトを出す
            if self.verbose:
                self.loss[i] = self.MSE(hx, y)

            # X_val入力ある場合MSEのリザルトをだす
            if type(X_val) == np.ndarray:
                val_pred = self._linear_hypothesis(X_val)
                self.val_loss[i] = self.MSE(val_pred, y_val)
