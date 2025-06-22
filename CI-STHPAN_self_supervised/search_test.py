import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
import torch.nn.functional as F
from tqdm import tqdm

# 1. **讀取 eod_data**
train_path = "/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/nyse_all/nyse_eod/eod_data_train.npy"
valid_path = "/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/nyse_all/nyse_eod/eod_data_valid.npy"

eod_train = np.load(train_path)  # (1026, 756, 5)
eod_valid = np.load(valid_path)  # (1026, 764, 5)
print(eod_train.shape)
print(eod_valid.shape)

# 2. **讀取 tickers**
tickers_path = "/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/src/data/datasets/stock/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv"
tickers = pd.read_csv(tickers_path, header=None).squeeze("columns").tolist()
assert len(tickers) == 1737, f"股票數量錯誤，應為 1026，實際為 {len(tickers)}"

# 3. **讀取轉折點**
turning_points_dir = "/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/nyse_all/nyse_turning/"
turning_points_dict = {
    ticker: np.load(f"{turning_points_dir}NYSE_{ticker}_turning_points.npy", allow_pickle=True).tolist()
    for ticker in tickers
}

# 4. **過濾掉超過 eod_train 長度 (755) 的轉折點**
max_time_index = eod_valid.shape[1]  # 756
filtered_turning_points_dict = {
    ticker: sorted([t for t in turning_points_dict[ticker] if t <= max_time_index - 1] + [max_time_index - 1])
    for ticker in tickers
}

def gpu_dtw(A, B):
    """使用 PyTorch GPU 計算 DTW"""
    if len(A) == 0 or len(B) == 0:
        return np.nan  # 避免空序列導致錯誤

    A_gpu = torch.tensor(A, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    B_gpu = torch.tensor(B, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        dtw_matrix = F.smooth_l1_loss(A_gpu, B_gpu, reduction='sum')  # 近似 DTW
    return dtw_matrix.item()

def generate_merged_intervals(turning_points):
    """
    依據轉折點產生合併區間，從相鄰轉折點開始，逐層合併，直到涵蓋所有轉折點
    """
    merged_intervals = []
    current_layer = [[turning_points[i], turning_points[i+1]] for i in range(len(turning_points)-1)]

    while len(current_layer) > 1:
        merged_intervals.append(current_layer)

        # **下一層合併**
        next_layer = []
        for i in range(0, len(current_layer) - 1, 2):
            merged_section = list(set(current_layer[i] + current_layer[i+1]))  # 合併相鄰兩組
            merged_section.sort()  # 確保順序正確
            next_layer.append(merged_section)

        # **如果剩下一個，直接加入**
        if len(current_layer) % 2 == 1:
            next_layer.append(current_layer[-1])

        current_layer = next_layer

    # 最後一層應該是完整轉折點範圍
    merged_intervals.append([turning_points])

    return merged_intervals
# **初始化 5 × 1026 × 1026 矩陣**
dtw_matrix = np.full((5, 1737, 1737), float("inf"))  # 初始值為無窮大

for feature_idx in range(5):  
    print(f"🔄 計算 Feature {feature_idx} 的 DTW 相似度...")

    for k in tqdm(range(1737), desc=f"Feature {feature_idx} - 股票迴圈"):
        benchmark_ticker = tickers[k]
        benchmark_turning_points = filtered_turning_points_dict[benchmark_ticker]

        if len(benchmark_turning_points) < 2:
            continue  # 如果轉折點太少，跳過

        merged_intervals = generate_merged_intervals(benchmark_turning_points)

        # 7. **計算 DTW**
        for intervals in merged_intervals:
            for interval in intervals:
                if len(interval) < 2:
                    continue  # 避免區間長度不足

                start, end = interval[0], interval[-1]  # **每個區間的起點 & 終點**
                if end >= eod_valid.shape[1]:  # 檢查索引是否超出範圍
                    continue

                # **取得基準股票 Feature 值**
                A_segment = eod_valid[k, start:end+1, feature_idx]

                if A_segment.shape[0] == 0:
                    continue  # 避免空序列錯誤
                max_val = np.max(A_segment)
                min_val = np.min(A_segment)
                range_percent = (max_val - min_val) / (min_val + 1e-8)  # 避免除 0
                                # **如果變動過小 (<10%)，則跳過**
                if range_percent < 0.1:
                    continue
                # **逐筆計算 GPU DTW**
                for i in range(1737):
                    if i == k:
                        dtw_matrix[feature_idx, k, i] = 0  # 自己的 DTW 設為 0
                        continue  # 跳過自己

                    B_segment = eod_valid[i, start:end+1, feature_idx]

                    if B_segment.shape[0] == 0:
                        continue  # 避免空序列錯誤

                    score = gpu_dtw(A_segment, B_segment)
                    interval_length = end - start + 1  # 計算區間長度
                    normalized_score = score / interval_length  # 標準化

                    # **更新最小 DTW 值**
                    dtw_matrix[feature_idx, k, i] = min(dtw_matrix[feature_idx, k, i], normalized_score)
                    #print(dtw_matrix)

# **儲存結果**
np.save("dtw_valid_matrix.npy", dtw_matrix)

print(f"✅ DTW 計算完成，結果已儲存至 dtw_valid_matrix")