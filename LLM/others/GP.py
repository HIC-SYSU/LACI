import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于脚本和服务器环境



class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}            ## 被固定，但实际可以优化 = = 》 最大化边缘对数似然来找到最优的参数
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        # self.is_fit = True

        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[
                1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()


# train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
# train_y = y(train_X, noise_sigma=1e-4)
# test_X = np.arange(0, 10, 0.1).reshape(-1, 1)
#
# gpr = GPR()
# gpr.fit(train_X, train_y)
# mu, cov = gpr.predict(test_X)
# test_y = mu.ravel()
# uncertainty = 1.96 * np.sqrt(np.diag(cov))
# plt.figure()
# plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
# plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
# plt.plot(test_X, test_y, label="predict")
# plt.scatter(train_X, train_y, label="train", c="red", marker="x")
# plt.legend()
# savefig("GP_is_fit.png")



# # fit GPR  ==== 库调用GP
# kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
# gpr.fit(train_X, train_y)
# mu, cov = gpr.predict(test_X, return_cov=True)
# test_y = mu.ravel()
# uncertainty = 1.96 * np.sqrt(np.diag(cov))
# # plotting
# plt.figure()
# plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
# plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
# plt.plot(test_X, test_y, label="predict")
# plt.scatter(train_X, train_y, label="train", c="red", marker="x")
# plt.legend()


def y_2d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y

train_X = np.random.uniform(-4, 4, (100, 2)).tolist()           ### 在[-4, 4]之间随机生成100个2维点
train_y = y_2d(train_X, noise_sigma=1e-4)

test_d1 = np.arange(-5, 5, 0.2)
test_d2 = np.arange(-5, 5, 0.2)
test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]

gpr = GPR(optimize=True)
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)
z = mu.reshape(test_d1.shape)

# fig = plt.figure(figsize=(7, 5))
# ax = Axes3D(fig)
# 创建图形和轴
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面、散点和等高线
surf = ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
scatter = ax.scatter(np.asarray(train_X)[:, 0], np.asarray(train_X)[:, 1], train_y, c=train_y, cmap=cm.coolwarm)
contour = ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)

# 设置标题和轴标签
ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Prediction')

# 保存图像
plt.savefig("3D_GPR_Result.png")
plt.close(fig)


