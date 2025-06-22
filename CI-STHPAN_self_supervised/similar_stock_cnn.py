import numpy as np
import numpy as np

import numpy as np
# 加載 .npy 文件
'''
data = np.load("/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/nyse_all/nyse_relation/dtw_valid_matrix.npy")

# 參數：選擇最小值的個數
k = 20

# 取得數據形狀 (5, 1026, 1026)
num_matrices, rows, cols = data.shape

# 創建一個全零矩陣作為輸出
binary_data = np.zeros_like(data)

# 對每個矩陣進行處理
for i in range(num_matrices):
    for j in range(cols):  # 只對每一列處理，避免重複
        col_indices = np.argpartition(data[i, :, j], k+1)[:k+1]  # 取出最小 k+1 個值的索引（包含對角線）
        binary_data[i, col_indices, j] = 1  # 設置這些位置為 1

# 保存處理後的數據
np.save("/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/nyse_all/nyse_relation/dtw_valid_matrix_binary.npy", binary_data)

print("處理完成，已保存為 binary .npy 文件，並包含對角線。")

'''
# 加載 .npy 文件
data = np.load("/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/nyse_all/nyse_relation/dtw_valid_matrix_binary.npy")
print(data)
# 選擇第一個矩陣 (5個中的第一個)
first_matrix = data[0]  # 形狀為 (1026, 1026)

# 取得第一行的數據
first_row = first_matrix[0]  # 形狀為 (1026,)

# 找到所有值為1的索引 (第一行)
row_indices = np.where(first_row == 1)[0]

# 計算第一行 1 的個數
num_row_ones = len(row_indices)

# 取得第一列的數據
first_col = first_matrix[:, 0]  # 形狀為 (1026,)

# 找到所有值為1的索引 (第一列)
col_indices = np.where(first_col == 1)[0]

# 計算第一列 1 的個數
num_col_ones = len(col_indices)

# 取得第二行的數據
second_row = first_matrix[1]  # 形狀為 (1026,)

# 找到所有值為1的索引 (第二行)
second_row_indices = np.where(second_row == 1)[0]

# 計算第二行 1 的個數
num_second_row_ones = len(second_row_indices)

# 取得第二列的數據
second_col = first_matrix[:, 1]  # 形狀為 (1026,)

# 找到所有值為1的索引 (第二列)
second_col_indices = np.where(second_col == 1)[0]

# 計算第二列 1 的個數
num_second_col_ones = len(second_col_indices)

# 輸出結果
print(f"第一行有 {num_row_ones} 個 1")
print(f"這些 1 位於索引: {row_indices}")
print(f"第一列有 {num_col_ones} 個 1")
print(f"這些 1 位於索引: {col_indices}")
print(f"第二行有 {num_second_row_ones} 個 1")
print(f"這些 1 位於索引: {second_row_indices}")
print(f"第二列有 {num_second_col_ones} 個 1")
print(f"這些 1 位於索引: {second_col_indices}")
