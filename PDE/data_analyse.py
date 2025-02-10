import numpy as np

# 读取 .npy 文件
data = np.load('city_1_predictions.npy')
last_12 = data[-12:, :, :]
new_data = np.concatenate((data, last_12), axis=0)

# 保存新的矩阵为 .npy 文件
np.save('NJ_PDE_all.npy', new_data)


# 打印数据的形状和类型
print("数据形状:", new_data.shape)
print("数据类型:", new_data.dtype)

data = np.load('city_2_predictions.npy')
last_12 = data[-12:, :, :]
new_data = np.concatenate((data, last_12), axis=0)

# 保存新的矩阵为 .npy 文件
np.save('CS_PDE_all.npy', new_data)

# 打印数据的形状和类型
print("数据形状:", new_data.shape)
print("数据类型:", new_data.dtype)
