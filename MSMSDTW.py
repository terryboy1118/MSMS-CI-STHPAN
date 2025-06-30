import os
import time
import numpy as np
from MSMSDTW_util import run_min_dtw, run_mean_dtw

def run_dtw_matrix(method: str, dataset: str, mode: str):
    """
    method: "min" or "mean"
    dataset: "nasdaq" or "nyse"
    mode: "train" or "valid"
    """
    print(f"\n🚀 開始計算 {method.upper()} 方法 - {dataset.upper()} - {mode} 模式")

    # 設定 eod_data 路徑
    base_path = "CI-STHPAN_self_supervised/src/data/datasets/stock"
    eod_path = f"{base_path}/{dataset}_all/eod_data_{dataset}/eod_data_{dataset}_{mode}.npy"
    eod_data = np.load(eod_path)
    print(eod_data.shape)
    # 執行相似度計算
    if method == "min":
        dtw_result = run_min_dtw(eod_data, dataset, mode)
    elif method == "mean":
        dtw_result = run_mean_dtw(eod_data, dataset, mode)
    else:
        raise ValueError("Unsupported method")

    # 儲存
    os.makedirs("results", exist_ok=True)
    output_path = f"/home/adam/CI-STHPAN-main/CI-STHPAN_self_supervised/src/data/datasets/stock/{dataset}_all/{dataset}_relation/dtw_{mode}_matrix_{method}_{dataset}.npy"

    np.save(output_path, dtw_result)
    print(f"✅ 儲存完成：{output_path}，shape: {dtw_result.shape}")

if __name__ == "__main__":
    methods = ["min", "mean"]
    datasets = ["nasdaq", "nyse"]
    modes = ["train", "valid"]

    for method in methods:
        for dataset in datasets:
            for mode in modes:
                output_path = f"results/dtw_{mode}_matrix_{method}_{dataset}.npy"
                if os.path.exists(output_path):
                    print(f"⚠️ 已存在，跳過：{output_path}")
                    continue

                start = time.time()
                run_dtw_matrix(method=method, dataset=dataset, mode=mode)
                print(f"⏱️ 耗時：{time.time() - start:.2f} 秒")
