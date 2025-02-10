# import numpy as np
# from scipy.optimize import minimize
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # 数据加载与划分
# # Y = np.load('Nanjing14days.npy')  # 加载包含 NaN 值的数据
# # Y_station = np.load('Nanjing_10_stations.npy')

# Y = np.load('Changshu14days.npy')  # 加载包含 NaN 值的数据
# # Y_station = np.load('Changshu_3_station.npy')

# # 确保数据中没有 NaN 值，处理方法：例如用0填充
# Y = np.nan_to_num(Y, nan=0.0)
# # Y_station = np.nan_to_num(Y_station, nan=0.0)

# train_data = Y[:20*12, :, :]  # 训练集
# val_data = Y[20*12:24*12, :, :]  # 验证集
# test_data = Y[336 - 4*12:, :, :]  # 测试集
# # test_data = Y_station[336 - 4*12:, :, :]

# # test_data = Y

# # test_data = Y[20*12:24*12, :, :]  # 验证集
# # val_data = Y[336 - 4*12:, :, :]  # 测试集

# # 假设 S[i, j] 需要通过优化来计算
# def optimize_S(historical_data):
#     """
#     使用凸优化从历史数据中学习 S[i, j]
#     historical_data: 包含历史数据的矩阵，形状为 (time_steps, 10, 10)
#     """
#     def objective(S_flat):
#         """
#         目标函数，通过最小化预测值与历史数据的误差来学习 S[i, j]
#         """
#         S = S_flat.reshape(10, 10)  # 将 S 从 1D 重塑为 2D
#         predicted_data = np.mean(historical_data + S, axis=0)  # 对历史数据和 S 进行求平均
#         error = np.sum((historical_data[-1, :, :] - predicted_data) ** 2)  # 计算预测误差
#         return error

#     # 使用历史数据的最后一个时间片来初始化 S
#     S_initial = np.zeros((10, 10))  # 假设初始 S 是零
    
#     # 使用优化方法（例如：BFGS）来最小化目标函数
#     result = minimize(objective, S_initial.flatten(), method='BFGS')
    
#     return result.x.reshape(10, 10)  # 返回学习到的 S

# # 计算 MAE, RMSE, MAPE
# def calculate_metrics(y_true, y_pred):
#     mask = y_true > 0  # 只计算真实值大于0的部分
#     mae = mean_absolute_error(y_true[mask], y_pred[mask])
#     rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
#     mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#     return mae, rmse, mape

# # 训练与预测过程
# def train_and_predict():
#     # 假设我们从训练集中学习 S[i, j]
#     S = optimize_S(train_data)  # 学习 S
#     print(S.shape)

#     # 预测未来12个时间片
#     predicted_data = []
#     for t in range(12, len(test_data)):
#         # 使用过去12个时间片数据来预测
#         # 此处使用预测公式，考虑到邻域点的影响
#         future_data = np.mean(
#             test_data[t-12:t, :, :] + np.roll(test_data[t-12:t, :, :], 1, axis=0) +  # 上
#             np.roll(test_data[t-12:t, :, :], -1, axis=0) +  # 下
#             np.roll(test_data[t-12:t, :, :], 1, axis=1) +  # 左
#             np.roll(test_data[t-12:t, :, :], -1, axis=1)   # 右
#         , axis=0) + S  # 加上通过优化学习的 S
#         predicted_data.append(future_data)
    
#     predicted_data = np.array(predicted_data)
    
#     # 测试阶段评估
#     true_y = test_data[12:, :, :]
#     pred_y = predicted_data
#     print(true_y.shape)
#     print(pred_y.shape)
    
#     # 计算 MAE, RMSE, MAPE
#     mae, rmse, mape = calculate_metrics(true_y, pred_y)
#     print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")

#     # last_12 = predicted_data[-12:, :, :]  # 取最后12个时间片
#     # predicted_data = np.concatenate([predicted_data, last_12], axis=0)
#     # # np.save('NJ_PDE.npy', predicted_data)  # 保存为.npy文件
#     # np.save('CS_PDE.npy', predicted_data)  # 保存为.npy文件


#     # print(predicted_data.shape)



# # 执行训练与预测
# train_and_predict()



# import numpy as np
# from scipy.optimize import minimize
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # 数据加载与划分
# def load_and_process_data(filenames):
#     """
#     加载多个数据文件，确保数据没有 NaN 值，并按时间顺序划分为训练集、验证集和测试集。
#     """
#     all_data = []
    
#     for filename in filenames:
#         data = np.load(filename)  # 加载数据
#         data = np.nan_to_num(data, nan=0.0)  # 替换 NaN 值
        
#         # 你可以根据实际需求来划分数据
#         train_data = data[:20*12, :, :]  # 训练集
#         val_data = data[20*12:24*12, :, :]  # 验证集
#         test_data = data[336 - 4*12:, :, :]  # 测试集
        
#         all_data.append((train_data, val_data, test_data))
    
#     return all_data

# # 假设 S[i, j] 需要通过优化来计算
# def optimize_S(historical_data):
#     """
#     使用凸优化从历史数据中学习 S[i, j]
#     historical_data: 包含历史数据的矩阵，形状为 (time_steps, 10, 10)
#     """
#     def objective(S_flat):
#         """
#         目标函数，通过最小化预测值与历史数据的误差来学习 S[i, j]
#         """
#         S = S_flat.reshape(10, 10)  # 将 S 从 1D 重塑为 2D
#         predicted_data = np.mean(historical_data + S, axis=0)  # 对历史数据和 S 进行求平均
#         error = np.sum((historical_data[-1, :, :] - predicted_data) ** 2)  # 计算预测误差
#         return error

#     # 使用历史数据的最后一个时间片来初始化 S
#     S_initial = np.zeros((10, 10))  # 假设初始 S 是零
    
#     # 使用优化方法（例如：BFGS）来最小化目标函数
#     result = minimize(objective, S_initial.flatten(), method='BFGS')
    
#     return result.x.reshape(10, 10)  # 返回学习到的 S

# # 计算 MAE, RMSE, MAPE
# def calculate_metrics(y_true, y_pred):
#     mask = y_true > 0  # 只计算真实值大于0的部分
#     mae = mean_absolute_error(y_true[mask], y_pred[mask])
#     rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
#     mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#     return mae, rmse, mape

# # 训练与预测过程
# def train_and_predict(all_data):
#     """
#     对多个城市的数据进行训练与预测。
#     """
#     # 假设我们从训练集中学习 S[i, j]
#     S = optimize_S(all_data[0][0])  # 使用第一个城市的训练数据学习 S
    
#     # 预测未来12个时间片
#     predicted_data_all_cities = []
    
#     for i, (train_data, val_data, test_data) in enumerate(all_data):
#         predicted_data = []
        
#         for t in range(12, len(test_data)):
#             # 使用过去12个时间片数据来预测
#             future_data = np.mean(
#                 test_data[t-12:t, :, :] + np.roll(test_data[t-12:t, :, :], 1, axis=0) +  # 上
#                 np.roll(test_data[t-12:t, :, :], -1, axis=0) +  # 下
#                 np.roll(test_data[t-12:t, :, :], 1, axis=1) +  # 左
#                 np.roll(test_data[t-12:t, :, :], -1, axis=1)   # 右
#             , axis=0) + S  # 加上通过优化学习的 S
#             predicted_data.append(future_data)
        
#         predicted_data = np.array(predicted_data)
#         predicted_data_all_cities.append(predicted_data)
        
#         # 计算评估指标
#         true_y = test_data[12:, :, :]
#         pred_y = predicted_data
#         print(f"City {i + 1} - true_y shape: {true_y.shape}, pred_y shape: {pred_y.shape}")
        
#         # 计算 MAE, RMSE, MAPE
#         mae, rmse, mape = calculate_metrics(true_y, pred_y)
#         print(f"City {i + 1} - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")
    
#     return predicted_data_all_cities


# # 文件路径
# filenames = ['Nanjing14days.npy', 'Changshu14days.npy', 'Shenzhen14days.npy', 'Sichuan14days.npy']  # 替换为实际的文件名

# # 加载数据
# all_data = load_and_process_data(filenames)

# # 训练与预测
# predicted_data_all_cities = train_and_predict(all_data)

# # 你可以根据需要将预测结果保存
# # np.save('predicted_data_all_cities.npy', predicted_data_all_cities) 

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 数据加载与划分
def load_and_process_data(filenames):
    """
    加载多个数据文件，确保数据没有 NaN 值，并按时间顺序划分为训练集、验证集和测试集。
    """
    all_data = []
    
    for filename in filenames:
        data = np.load(filename)  # 加载数据
        data = np.nan_to_num(data, nan=0.0)  # 替换 NaN 值
        
        # 你可以根据实际需求来划分数据
        train_data = data[:20*12, :, :]  # 训练集
        val_data = data[20*12:24*12, :, :]  # 验证集
        # test_data = data[336 - 4*12:, :, :]  # 测试集
        test_data = data

        
        all_data.append((train_data, val_data, test_data))
    
    return all_data

# 假设 S[i, j] 需要通过优化来计算
def optimize_S(all_train_data):
    """
    使用凸优化从多个城市的训练数据中学习共享的 S[i, j]
    all_train_data: 包含所有城市训练数据的列表，每个元素是 (time_steps, 10, 10) 形状的训练数据
    """
    def objective(S_flat):
        """
        目标函数，通过最小化预测值与历史数据的误差来学习 S[i, j]
        """
        S = S_flat.reshape(10, 10)  # 将 S 从 1D 重塑为 2D
        error = 0
        
        # 对每个城市的训练数据进行计算
        for train_data in all_train_data:
            predicted_data = np.mean(train_data + S, axis=0)  # 对历史数据和 S 进行求平均
            error += np.sum((train_data[-1, :, :] - predicted_data) ** 2)  # 计算误差

        return error

    # 使用所有城市的训练数据来初始化 S
    S_initial = np.zeros((10, 10))  # 假设初始 S 是零
    
    # 使用优化方法（例如：BFGS）来最小化目标函数
    result = minimize(objective, S_initial.flatten(), method='BFGS')
    
    return result.x.reshape(10, 10)  # 返回学习到的 S

# 计算 MAE, RMSE, MAPE
def calculate_metrics(y_true, y_pred):
    mask = y_true > 0  # 只计算真实值大于0的部分
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, mape

# 训练与预测过程
def train_and_predict(all_data):
    """
    对多个城市的数据进行训练与预测，并将预测结果分别保存为CSV文件。
    """
    # 使用所有城市的训练数据学习共享的 S[i, j]
    all_train_data = [train_data for train_data, _, _ in all_data]  # 从每个城市获取训练数据
    S = optimize_S(all_train_data)  # 使用四个城市的训练数据来学习 S
    
    predicted_data_all_cities = []
    
    for i, (train_data, val_data, test_data) in enumerate(all_data):
        predicted_data = []
        
        for t in range(12, len(test_data)):
            # 使用过去12个时间片数据来预测
            future_data = np.mean(
                test_data[t-12:t, :, :] + np.roll(test_data[t-12:t, :, :], 1, axis=0) +  # 上
                np.roll(test_data[t-12:t, :, :], -1, axis=0) +  # 下
                np.roll(test_data[t-12:t, :, :], 1, axis=1) +  # 左
                np.roll(test_data[t-12:t, :, :], -1, axis=1)   # 右
            , axis=0) + S  # 加上通过优化学习的 S
            predicted_data.append(future_data)
        
        predicted_data = np.array(predicted_data)
        predicted_data_all_cities.append(predicted_data)
        
        # 计算评估指标
        true_y = test_data[12:, :, :]
        pred_y = predicted_data
        print(f"City {i + 1} - true_y shape: {true_y.shape}, pred_y shape: {pred_y.shape}")
        
        # 计算 MAE, RMSE, MAPE
        mae, rmse, mape = calculate_metrics(true_y, pred_y)
        print(f"City {i + 1} - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")
        
        # 保存预测结果到CSV文件
        save_to_npy(predicted_data, i + 1)

    return predicted_data_all_cities

# 保存预测结果为Numpy文件
def save_to_npy(predicted_data, city_index):
    """
    将预测结果保存为Numpy文件，每个城市一个文件。
    """
    # 保存预测数据到 .npy 文件
    filename = f"city_{city_index}_predictions.npy"
    np.save(filename, predicted_data)
    print(f"City {city_index} predictions saved to {filename}")


# 文件路径
# filenames = ['city1.npy', 'city2.npy', 'city3.npy', 'city4.npy']  # 替换为实际的文件名
filenames = ['Nanjing14days.npy', 'Changshu14days.npy', 'Shenzhen14days.npy', 'Sichuan14days.npy']  # 替换为实际的文件名


# 加载数据
all_data = load_and_process_data(filenames)

# 训练与预测
predicted_data_all_cities = train_and_predict(all_data)

# 你可以根据需要将预测结果保存
# np.save('predicted_data_all_cities.npy', predicted_data_all_cities) 


