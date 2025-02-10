'''
Data pre-process
[336, 10, 10] --> [336, 100]
' _reshaped.npy'
' _processed_reshaped.npy' miss_fill_as_average
'''
# import numpy as np

# # 加载数据
# # data = np.load("AirData/Changshu14days.npy")
# data = np.load("AirData/Nanjing14days.npy")

# print("Original shape:", data.shape)
# print("Initial NaN values:", type(data))

# # 第一步：使用空间相邻值填充
# for m in range(data.shape[0]):
#     for i in range(data.shape[1]):
#         for j in range(data.shape[2]):
#             if np.isnan(data[m, i, j]):
#                 neighbors = []
#                 # 检查上下左右相邻位置
#                 if i > 0 and not np.isnan(data[m, i-1, j]):
#                     neighbors.append(data[m, i-1, j])
#                 if i < data.shape[1]-1 and not np.isnan(data[m, i+1, j]):
#                     neighbors.append(data[m, i+1, j])
#                 if j > 0 and not np.isnan(data[m, i, j-1]):
#                     neighbors.append(data[m, i, j-1])
#                 if j < data.shape[2]-1 and not np.isnan(data[m, i, j+1]):
#                     neighbors.append(data[m, i, j+1])
                
#                 if neighbors:
#                     data[m, i, j] = np.mean(neighbors)

# print("NaN values after spatial filling:", np.isnan(data).sum())

# # 第二步：使用时间维度上的均值填充剩余的NaN值
# for i in range(data.shape[1]):
#     for j in range(data.shape[2]):
#         mask = np.isnan(data[:, i, j])
#         if np.any(mask):  # 如果这个位置上还有NaN值
#             valid_values = data[:, i, j][~mask]  # 获取非NaN值
#             if len(valid_values) > 0:  # 如果存在有效值
#                 mean_value = np.mean(valid_values)
#                 data[:, i, j][mask] = mean_value



# # 重塑数据
# reshaped_data = data.reshape(336, -1)
# print("New shape:", reshaped_data.shape)

# # np.save("Changshu14days_processed_reshaped.npy", reshaped_data)
# np.save("Naning14days_processed_reshaped.npy", reshaped_data)
# print("Final NaN values:", np.isnan(reshaped_data).sum())

# # Orig_data = np.load("AirData/Changshu14days.npy")
# Orig_data = np.load("AirData/Nanjing14days.npy")

# Orig_reshaped_data = Orig_data.reshape(336, -1)
# # np.save("Changshu14days_reshaped.npy", Orig_reshaped_data)
# np.save("Nanjing14days_reshaped.npy", Orig_reshaped_data)

# print("Final NaN values:", np.isnan(Orig_reshaped_data).sum())

'''
.npy --> .csv
'''
# import numpy as np
# import pandas as pd

# # 加载 .npy 文件
# data = np.load("Nanjing14days_reshaped.npy")

# # 将数据转换为 DataFrame
# df = pd.DataFrame(data)

# # 保存为 .csv 文件
# df.to_csv("Nanjing14days_reshaped.csv", index=False)

# print("转换完成，文件已保存为 Nanjing14days_reshaped.csv")

'''
datetime --> CSDI datetime type
'''

# import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv("Nanjing14days_reshaped.csv")

# # 假设第一列是日期列，首先将其转换为 datetime 格式
# df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

# # 格式化日期为 'YYYY/MM/DD HH:MM:SS'
# df.iloc[:, 0] = df.iloc[:, 0].dt.strftime('%Y/%m/%d %H:%M:%S')

# # 保存为新的 CSV 文件
# df.to_csv("Nanjing14days.csv", index=False)

# print("日期格式已统一，文件已保存为 Nanjing14days_reshaped_formatted.csv")

'''
.csv --> .txt
'''

# import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv("Nanjing14days.csv")

# # 保存为 .txt 文件
# # 使用制表符作为分隔符，或者你可以选择其他分隔符
# df.to_csv("Nanjing14days.txt", sep=',', index=False)

# print("文件已保存为 Nanjing4days.txt")

'''
数据填充
'''
import numpy as np

# 读取 .npy 文件
# data = np.load('AirData/Nanjing14days.npy')  # 将 'your_file.npy' 替换为你的文件路径
# data = np.load('AirData/Changshu14days.npy')  # 将 'your_file.npy' 替换为你的文件路径
# data = np.load('CS_PDE.npy')  # 将 'your_file.npy' 替换为你的文件路径
data = np.load('NJ_PDE.npy')  # 将 'your_file.npy' 替换为你的文件路径




# 获取数据的形状
n, x_size, y_size = data.shape

# 遍历所有的 (i, x, y)，填充 NaN 值
for i in range(n):  # 从第0帧开始
    for x in range(x_size):
        for y in range(y_size):
            if np.isnan(data[i, x, y]):  # 如果当前位置是 NaN
                # 获取前12帧的数据
                start_idx = max(0, i-12)
                slice_data = data[start_idx:i, x, y]
                
                # 过滤掉 NaN 值并检查切片是否为空
                valid_values = slice_data[~np.isnan(slice_data)]
                
                if valid_values.size > 0:  # 如果切片中有有效值
                    # 计算前12帧 (i-12:i) 在 (x, y) 位置的均值（忽略 NaN）
                    data[i, x, y] = np.nanmean(valid_values)
                else:
                    # 如果前12帧都没有有效数据，则用当前帧 (i, :, :) 的均值填充
                    # 检查当前帧是否有有效数据，如果没有，则用一个默认值填充（例如0）
                    current_frame_mean = np.nanmean(data[i, :, :])
                    if np.isnan(current_frame_mean):  # 如果当前帧也全是 NaN
                        data[i, x, y] = 0  # 用0或其他值填充
                    else:
                        data[i, x, y] = current_frame_mean

# 保存修改后的数据（可选）
# np.save('Nanjing_nozero.npy', data)
# np.save('Changshu_PDE_process.npy', data)
np.save('Nanjing_PDE_process.npy', data)



# 重新加载保存后的数据
# data = np.load('Nanjing_nozero.npy')  # 将 'filled_data.npy' 替换为你的文件路径
# data = np.load('Changshu_PDE_process.npy')  # 将 'filled_data.npy' 替换为你的文件路径
data = np.load('Nanjing_PDE_process.npy')  # 将 'filled_data.npy' 替换为你的文件路径


# 打印数据中的第一帧，检查是否有 NaN
print(data[0, :, :])

# 检查数据中是否仍然包含 NaN 值
if np.isnan(data).any():
    print("数据中仍然包含 NaN 值")
else:
    print("数据中没有 NaN 值")
