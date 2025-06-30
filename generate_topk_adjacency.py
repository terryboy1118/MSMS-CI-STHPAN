import numpy as np
import os

def generate_binary_topk_matrix(dtw_matrix, k=20):
    binary_matrix = np.zeros_like(dtw_matrix)
    num_features, N, _ = dtw_matrix.shape

    for i in range(num_features):
        for col in range(N):
            topk_idx = np.argpartition(dtw_matrix[i, :, col], k + 1)[:k + 1]
            binary_matrix[i, topk_idx, col] = 1

    return binary_matrix

def inspect_binary_matrix(binary_matrix, feature_index=0, row_idx=0, col_idx=0):
    print(f"\U0001F4DC 檢查 Feature {feature_index}：")

    # 檢查 row
    row_ones = np.where(binary_matrix[feature_index, row_idx] == 1)[0]
    print(f"第 {row_idx} 行有 {len(row_ones)} 個 1，索引位置：{row_ones.tolist()}")

    # 檢查 column
    col_ones = np.where(binary_matrix[feature_index, :, col_idx] == 1)[0]
    print(f"第 {col_idx} 列有 {len(col_ones)} 個 1，索引位置：{col_ones.tolist()}")

if __name__ == "__main__":
    base_dir = "/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/src/data/datasets/stock"
    methods = ["min", "mean"]
    datasets = ["nyse", "nasdaq"]
    modes = ["train", "valid"]

    for method in methods:
        for dataset in datasets:
            for mode in modes:
                input_path = f"{base_dir}/{dataset}_all/{dataset}_relation/dtw_{mode}_matrix_{method}_{dataset}.npy"
                output_path = f"{base_dir}/relation/MSMSDTW-{method}/dtw_{mode}_matrix_{method}_{dataset}_binary.npy"

                if not os.path.exists(input_path):
                    print(f"❌ 檔案不存在，略過：{input_path}")
                    continue

                print(f"📁 處理檔案：{input_path}")
                dtw_matrix = np.load(input_path)
                binary_matrix = generate_binary_topk_matrix(dtw_matrix, k=20)
                np.save(output_path, binary_matrix)
                print(f"✅ 已儲存 binary 矩陣至 {output_path}")

                # 可選：僅檢查第一組的數據
                if method == "min" and dataset == "nyse" and mode == "train":
                    inspect_binary_matrix(binary_matrix, feature_index=0, row_idx=0, col_idx=0)
                    inspect_binary_matrix(binary_matrix, feature_index=0, row_idx=1, col_idx=1)
