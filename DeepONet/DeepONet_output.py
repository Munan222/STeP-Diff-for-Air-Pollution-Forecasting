"""
Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from mask_metric import masked_mae,masked_mape,masked_rmse,masked_mse


data_orig = np.load("AirData/Nanjing14days.npy")
data_orig = np.load("AirData/Changshu14days.npy")


data_orig = np.nan_to_num(data_orig, nan=0)
data_station= np.load("AirData/Nanjing_10_stations.npy")
# data_station = np.load("AirData/Changshu_3_station.npy")
data_station = np.nan_to_num(data_station, nan=0)

# data = np.load("Nanjing_nozero.npy")
# data = np.load("Changshu_nozero.npy")
# data = np.load("Changshu_PDE_process.npy")
data = np.load("Nanjing_PDE_process.npy")




'''
Nanjing
'''
'''
(28,1200)
'''
data = data.reshape(-1, 12, 10, 10).reshape(-1, 1200)
data_orig = data_orig.reshape(-1, 12, 10, 10).reshape(-1, 1200)
data_station = data_station.reshape(-1, 12, 10, 10).reshape(-1, 1200)

# print(data.shape)

train_x = data[0:18, :]
train_y = data[1:19, :]  # changshu

# val_x = data_fill['val_x'].reshape(-1, 12, 10, 10).reshape(-1, 1200)
# val_y = data_fill['val_y'].reshape(-1, 12, 10, 10).reshape(-1, 1200)

# test_x = data[20:26, :]
# test_y = data_orig[21:27, :]

# test_x = data[0:28, :]
# test_y = data_orig[0:28, :]

# test_x = data[20:26, :]
test_x = data
# test_y = data[21:27, :]
test_y = data

# test_y = data_station[21:27, :]


print(test_x.shape)
# test_x = data[20:21, :]
# test_y = data_orig[21:22, :]

# test_x = data[25:26, :]
# test_y = data_orig[26:27, :]

'''
(336,100)
'''
# data = data.reshape(-1, 100)
# data_orig = data_orig.reshape(-1, 100)


# train_x = data[0:19*12-1, :]
# train_y = data[1*12:20*12-1, :] 

# test_x = data[20*12:27*12-1, :]
# test_y = data_orig[21*12:28*12-1, :]

'''
Changshu
'''
# train_x = data['train_x'].reshape(187, -1)
# train_y = data['train_y'].reshape(187, -1)

# val_x = data['val_x'].reshape(62, -1)
# val_y = data['val_y'].reshape(62, -1)

# test_x = data['test_x'].reshape(62, -1)
# test_y = data['test_y'].reshape(62, -1)

'''
Type 1
'''
'''Trunk {1-100}'''
# X_train = (data[0:222, :].astype(np.float32), np.arange(1, 101).reshape(100, 1).astype(np.float32))
# y_train = data[12:234, :].astype(np.float32)

# X_test = (data[235:256, :].astype(np.float32), np.arange(1, 101).reshape(100, 1).astype(np.float32))
# y_test = data[247:268, :].astype(np.float32)

# X_val= (data[269:323, :].astype(np.float32), np.arange(1, 101).reshape(100, 1).astype(np.float32))
# y_val = data_orig[281:335, :].astype(np.float32)


'''Trunk {1}*100'''
# X_train = (data[0:222, :].astype(np.float32), np.ones((100, 1), dtype=np.float32))
# X_test = (data[235:256, :].astype(np.float32), np.ones((100, 1), dtype=np.float32))
# X_val = (data[269:323, :].astype(np.float32), np.ones((100, 1), dtype=np.float32))
# y_train = data[12:234, :].astype(np.float32)
# y_test = data[247:268, :].astype(np.float32)
# y_val = data_orig[281:335, :].astype(np.float32)

'''
Type 2
'''
'''Trunk {1-100}'''
# X_train = (data[0:117, :].astype(np.float32), np.arange(1, 101).reshape(100, 1).astype(np.float32))
# y_train = data[117:234, :].astype(np.float32)

# X_test = (data[235:247, :].astype(np.float32), np.arange(1, 101).reshape(100, 1).astype(np.float32))
# y_test = data[248:260, :].astype(np.float32)

# X_val= (data[269:281, :].astype(np.float32), np.arange(1, 101).reshape(100, 1).astype(np.float32))
# y_val = data_orig[282:294, :].astype(np.float32)


# num = np.arange(1, 101)

# repeated_values = np.repeat(num, 12)

# trunk = repeated_values.reshape(1200, 1)

seq_len = 1200  # 序列长度
# seq_len = 100  # 序列长度

encoding_dim = 1

def get_positional_encoding(seq_len, encoding_dim):
    """
    生成形状为 (seq_len, encoding_dim) 的位置编码。
    由于目标是生成 (1200, 1)，我们将通过正弦计算得到单一维度的编码。
    """
    position = np.arange(0, seq_len).reshape(-1, 1)  # 生成位置索引 (seq_len, 1)

    # 初始化位置编码矩阵
    pe = np.zeros((seq_len, encoding_dim))

    # 使用正弦函数计算位置编码
    pe[:, 0] = np.sin(position[:, 0] / 10000.0)  # 对每个位置计算 sin 值

    return pe  # 返回位置编码矩阵

# 获取形状为 (1200, 1) 的位置编码
position_encoding = get_positional_encoding(seq_len, encoding_dim)

trunk = position_encoding

X_train = (train_x, trunk)
y_train = train_y

X_test = (test_x, trunk)
y_test = test_y

# X_val= (val_x, trunk)
# y_val = val_y


# print(X_test[0].shape, X_val[0].shape)


'''Trunk {1}*100'''
# X_train = (data[0:117, :].astype(np.float32), np.ones((100, 1), dtype=np.float32))
# X_test = (data[235:247, :].astype(np.float32), np.ones((100, 1), dtype=np.float32))
# X_val = (data[269:281, :].astype(np.float32), np.ones((100, 1), dtype=np.float32))
# y_train = data[117:234, :].astype(np.float32)
# y_test = data[248:260, :].astype(np.float32)
# y_val = data_orig[282:294, :].astype(np.float32)


# # Load dataset
# d = np.load("deeponet_antiderivative_aligned/antiderivative_aligned_test.npz", allow_pickle=True)
# X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
# y_train = d["y"].astype(np.float32)
# d = np.load("deeponet_antiderivative_aligned/antiderivative_aligned_test.npz", allow_pickle=True)
# X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
# y_test = d["y"].astype(np.float32)

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# Choose a network
# m = 100
m = 1200
dim_x = 1

net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"]) # changshu
# model.compile("adam", lr=0.005, metrics=["mean l2 relative error"])
# model.compile("adam", lr=0.01, metrics=["mean l2 relative error"])
# model.compile("adam", lr=0.1, metrics=["mean l2 relative error"])


losshistory, train_state = model.train(iterations=10000)
# losshistory, train_state = model.train(iterations=5000)


# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()

y_test_pred = model.predict(X_test)
# y_val_pred = model.predict(X_val)

def calculate_metrics(y_true, y_pred):

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    mae,mape,rmse = masked_mae(y_pred_filtered, y_true_filtered,0.0),masked_mape(y_pred_filtered, y_true_filtered,0.0),masked_rmse(y_pred_filtered, y_true_filtered,0.0)
    print('Test_MAE: {}, Test_RMSE: {}, Test_MAPE: {}'.format(mae, rmse, mape))

    return mae, rmse, mape
    

    

# print("val")
# mae_val, rmse_val, mape_val = calculate_metrics(y_val, y_val_pred)
print("Test")
mae_test, rmse_test, mape_test = calculate_metrics(y_test, y_test_pred)

# print(y_test.shape)
# print(y_test_pred.shape)

test_results_df = pd.DataFrame({
    'Test Actual': y_test.flatten(),
    'Test Predicted': y_test_pred.flatten()
})


metrics_df = pd.DataFrame({
    'Metric': ['MAE test', 'RMSE test', 'MAPE test'],
    'Value': [mae_test, rmse_test, mape_test]
})

test_results_df.to_csv('Results/NJ_test_prediction_results.csv', index=False)
metrics_df.to_csv('Results/NJ_evaluation_metrics.csv', index=False)

# test_results_df.to_csv('Results/CS_test_prediction_results.csv', index=False)
# metrics_df.to_csv('Results/CS_evaluation_metrics.csv', index=False)



pred =  y_test_pred.reshape(-1, 12, 10, 10).reshape(-1, 10, 10)

print(pred.shape)

np.save('NJ_data_deeponet.npy', pred)
# np.save('CS_data_deeponet.npy', pred)
