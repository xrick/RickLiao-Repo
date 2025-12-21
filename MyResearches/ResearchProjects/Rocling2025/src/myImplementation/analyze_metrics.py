import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def analyze_gop_metrics(csv_filepath: str):
    """
    載入GOP指標CSV檔案，計算每個指標在發音錯誤檢測上的分類效能。
    
    這個函數會遵循 Parikh et al. (2025) 的方法，為每個指標尋找
    能夠最大化MCC分數的最佳分類門檻值，然後回報在該門檻值下的各項效能指標。
    
    Args:
        csv_filepath (str): 包含GOP指標和'mispronounced'標籤的CSV檔案路徑。
    """
    try:
        df = pd.read_csv(csv_filepath)
        print(f"成功載入檔案: {csv_filepath}")
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 '{csv_filepath}'。請確認檔案名稱和路徑是否正確。")
        return

    # 'high'表示分數越高，錯誤發音的可能性越大
    # 'low'表示分數越低，錯誤發音的可能性越大
    metrics_to_evaluate = {
        # 基線方法
        'max_logit': 'low',
        'mean_logit_margin': 'low',
        'prosetrior_probability': 'low',
        'logit_variance': 'high',
        # 我們提出的方法
        'evt_k3': 'low',
        'skewness': 'high',
        'kurtosis': 'high',
        'autocorr_lag1': 'low',
        'entropy_mean': 'high',
        'kl_to_onehot': 'high',
        # GMM 特徵
        'gmm_means_0': 'low',
        'gmm_means_1': 'low',
        'gmm_vars_0': 'high',
        'gmm_vars_1': 'high',
        'gmm_weights_0': 'high',
        'gmm_weights_1': 'high',
    }

    results = []

    for metric, direction in metrics_to_evaluate.items():
        if metric not in df.columns:
            continue

        subset = df[[metric, 'mispronounced']].dropna()
        y_true_subset = subset['mispronounced']
        scores = subset[metric]

        if len(scores) == 0:
            continue

        thresholds = np.percentile(scores, np.arange(0, 101, 1))
        best_mcc = -1
        best_threshold = None

        for threshold in np.unique(thresholds): # 使用 unique 避免重複計算
            y_pred = scores < threshold if direction == 'low' else scores > threshold
            mcc = matthews_corrcoef(y_true_subset, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

        if best_threshold is not None:
            final_y_pred = scores < best_threshold if direction == 'low' else scores > best_threshold
            
            results.append({
                'Method': metric,
                'Accuracy': accuracy_score(y_true_subset, final_y_pred),
                'Precision': precision_score(y_true_subset, final_y_pred, zero_division=0),
                'Recall': recall_score(y_true_subset, final_y_pred, zero_division=0),
                'F1-Score': f1_score(y_true_subset, final_y_pred, zero_division=0),
                'MCC': best_mcc
            })

    results_df = pd.DataFrame(results).sort_values(by='MCC', ascending=False)
    
    print("\n" + "="*80)
    print("       發音錯誤檢測之分類效能比較 (以 MCC 分數排序)")
    print("="*80)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("="*80)

if __name__ == '__main__':
    # 將 'myimpl_so_metrics.csv' 替換成您的檔案實際名稱
    # Ensure the CSV file is in the same directory as this script,
    # or provide the full path to it.
    csv_file = 'myimpl_so_metrics.csv'
    analyze_gop_metrics(csv_file)