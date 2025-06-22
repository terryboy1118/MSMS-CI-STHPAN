import numpy as np
import matplotlib.pyplot as plt
import os

# 載入 .npy 檔案
data = np.load('/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/src/data/datasets/stock/relation/dtw/NASDAQ_dtw_valid_fine_top20.npy')
print('資料形狀:', data.shape)  # (5, 1026, 1026)

# 取出第5個矩陣
matrix = data[4]
print('取出的矩陣形狀:', matrix.shape)

# 建立儲存資料夾
save_dir = '/home/adam/CI-STHPAN-main/nasdaq_all/RELATION/heatmap'
os.makedirs(save_dir, exist_ok=True)

# 畫圖（白底黑點）
plt.figure(figsize=(12, 12))
plt.imshow(matrix, cmap='Greys', interpolation='none')  # 白底黑點呈現
plt.title('Binary Relation Matrix (1 = black)', fontsize=16)
plt.xlabel('Stock j', fontsize=12)
plt.ylabel('Stock i', fontsize=12)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.tight_layout()

# 存檔
save_path = os.path.join(save_dir, 'relation_matrix_binary_clean_original.png')
plt.savefig(save_path, dpi=300)
print(f"已儲存至: {save_path}")
plt.close()
