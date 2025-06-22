import numpy as np

# 載入資料
data = np.load('/home/adam/CI-STHPAN-main/nasdaq_all/RELATION/gpu_dtw_similarity_matrix_train_binary.npy')
# 這次換從「列」變成「行」：
matrix = data[4]  # 你的第五個矩陣
for stock_idx in range(10):  # 只做前10個股票（第0到第9個）
    selected_by_indices = np.where(matrix[:, stock_idx] == 1)[0]  # 注意這邊是[:, stock_idx]，看「直的」
    print(f"股票 {stock_idx} 被這些股票選了：{selected_by_indices.tolist()}")
col_sums = np.sum(matrix, axis=0)


# 印出統計資訊
print("每列被選為關聯對象的次數統計：")
print(f"最小值: {np.min(col_sums)}")
print(f"最大值: {np.max(col_sums)}")
print(f"平均值: {np.mean(col_sums):.2f}")

# 找出被選超過 30 次的股票 index（範例）
popular_stocks = np.where(col_sums > 30)[0]
print(f"被選超過 30 次的股票 index: {popular_stocks}")
