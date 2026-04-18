import numpy as np
import GPy

# 创建示例数据，两个形状为 (4, 32, 8, 64, 64) 的张量
tensor1 = np.random.rand(4, 32, 8, 64, 64)
tensor2 = np.random.rand(4, 32, 8, 64, 64)
tensor3 = np.random.rand(4, 32, 8, 64, 64)
X = tensor1.reshape(tensor1.shape[0], -1)
Y = tensor2.reshape(tensor2.shape[0], -1)
X_new = tensor3.reshape(tensor2.shape[0], -1)


# # 定义高斯过程模型
# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# model = GPy.models.GPRegression(X, Y, kernel)
# # 训练模型
# model.optimize(messages=True)
# # 使用模型进行预测
# Y_pred, Y_var = model.predict(X)
# # 输出训练的模型参数
# print(model)
# print("Y_pred:", Y_pred)
# print("Y_var:", Y_var)


# let X, Y be data loaded above
# Model creation:
m = GPy.models.GPRegression(X, Y)
m.optimize()
# 1: Saving a model:
np.save('model_save.npy', m.param_array)
# 2: loading a model
# Model creation, without initialization:
m_load = GPy.models.GPRegression(X, Y, initialize=False)
m_load.update_model(False) # do not call the underlying expensive algebra on load
m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
m_load[:] = np.load('model_save.npy') # Load the parameters
m_load.update_model(True) # Call the algebra only once
print(m_load)

# 进行预测
Y_pred, Y_var = m_load.predict(X)
Y_pred = Y_pred.reshape(4, 32, 8, 64, 64)
# 输出预测结果
print("Predicted means:", Y_pred-tensor2)
print("Predicted variances:", Y_var)