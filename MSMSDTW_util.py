import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def gpu_dtw(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan

    A_gpu = torch.tensor(A, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    B_gpu = torch.tensor(B, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        dtw_matrix = F.smooth_l1_loss(A_gpu, B_gpu, reduction='sum')
    return dtw_matrix.item()

def generate_merged_intervals(turning_points):
    merged_intervals = []
    current_layer = [[turning_points[i], turning_points[i+1]] for i in range(len(turning_points)-1)]

    while len(current_layer) > 1:
        merged_intervals.append(current_layer)
        next_layer = []
        for i in range(0, len(current_layer) - 1, 2):
            merged_section = list(set(current_layer[i] + current_layer[i+1]))
            merged_section.sort()
            next_layer.append(merged_section)
        if len(current_layer) % 2 == 1:
            next_layer.append(current_layer[-1])
        current_layer = next_layer

    merged_intervals.append([turning_points])
    return merged_intervals

def run_min_dtw(eod_data, dataset, mode):
    return _run_dtw_core(eod_data, dataset, mode, method="min")

def run_mean_dtw(eod_data, dataset, mode):
    return _run_dtw_core(eod_data, dataset, mode, method="mean")

def _run_dtw_core(eod_data, dataset, mode, method="min"):
    base_path = "CI-STHPAN_self_supervised/src/data/datasets/stock"
    turning_dir = f"{base_path}/{dataset}_all/{dataset}_turning_points/"
    tickers_path = f"{base_path}/{dataset.upper()}_tickers_qualify_dr-0.98_min-5_smooth.csv"

    tickers = pd.read_csv(tickers_path, header=None).squeeze("columns").tolist()

    turning_points_dict = {
        ticker: np.load(f"{turning_dir}{dataset.upper()}_{ticker}_turning_points.npy", allow_pickle=True).tolist()
        for ticker in tickers
    }

    max_time_index = eod_data.shape[1]
    filtered_turning_points_dict = {
        ticker: sorted([t for t in turning_points_dict[ticker] if t <= max_time_index - 1] + [max_time_index - 1])
        for ticker in tickers
    }

    dtw_matrix = np.full((5, len(tickers), len(tickers)), float("inf"))

    for feature_idx in range(5):
        print(f"🔄 Feature {feature_idx} - {dataset.upper()} {mode} ({method})")

        for k in tqdm(range(len(tickers)), desc=f"Feature {feature_idx} - Base Loop"):
            benchmark_ticker = tickers[k]
            benchmark_turning_points = filtered_turning_points_dict[benchmark_ticker]

            if len(benchmark_turning_points) < 2:
                continue

            merged_intervals = generate_merged_intervals(benchmark_turning_points)

            count_matrix = np.zeros((len(tickers), len(tickers))) if method == "mean" else None

            for intervals in merged_intervals:
                for interval in intervals:
                    if len(interval) < 2:
                        continue

                    start, end = interval[0], interval[-1]
                    if end >= eod_data.shape[1]:
                        continue

                    A_segment = eod_data[k, start:end+1, feature_idx]
                    if A_segment.shape[0] == 0:
                        continue
                    max_val, min_val = np.max(A_segment), np.min(A_segment)
                    if (max_val - min_val) / (min_val + 1e-8) < 0.1:
                        continue

                    for i in range(len(tickers)):
                        if i == k:
                            dtw_matrix[feature_idx, k, i] = 0
                            continue

                        B_segment = eod_data[i, start:end+1, feature_idx]
                        if B_segment.shape[0] == 0:
                            continue

                        score = gpu_dtw(A_segment, B_segment)
                        normalized_score = score / (end - start + 1)

                        if method == "min":
                            dtw_matrix[feature_idx, k, i] = min(dtw_matrix[feature_idx, k, i], normalized_score)
                        elif method == "mean":
                            if np.isinf(dtw_matrix[feature_idx, k, i]):
                                dtw_matrix[feature_idx, k, i] = 0
                            dtw_matrix[feature_idx, k, i] += normalized_score
                            if not np.isnan(normalized_score):
                                count_matrix[k, i] += 1

            if method == "mean":
                for i in range(len(tickers)):
                    if count_matrix[k, i] > 0:
                        dtw_matrix[feature_idx, k, i] /= count_matrix[k, i]

        torch.cuda.empty_cache()

    return dtw_matrix
