import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import os

# ------------------------------------------------------
# Configuration parameters (modify paths as needed)
# ------------------------------------------------------
FILE_PATHS = [  # 12 CSV file paths
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_cnn_only_DO/CNN_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_LSTM_DO/LSTM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_BiLSTM_DO/BiLSTM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_Cnn_fgsm_sam_DO/cnn_fgsm_sam_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_LSTM_FGSM_SAM_DO/LSTM_FGSM_SAM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_BiLSTM_fgsm-sam_DO/BiLSTM_fgsm-sam_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_CNN_TN/CNN_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_LSTM_TN/LSTM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_BiLSTM_TN/BiLSTM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_CNN_FGSM_SAM_only_TN/CNN_FGSM_SAM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_LSTM_FGSM_SAM_TN/LSTM_FGSM_curve_comparison.csv",
    "C:/Users/LEGION/Desktop/fsdownload/sandiantu/output_BiLSTM_FGSM_SAM_TN/BiLSTM_FGSM_SAM_curve_comparison.csv"
]

FIGSIZE = (12, 8)
DPI = 300
SAVE_PATH = "twelve_subplots_result.png"
POINT_SIZE = 5
ALPHA = 0.6
COLOR = "blue"

# ------------------------------------------------------
# Core: Read file and extract data (auto-detect column names)
# ------------------------------------------------------
def load_data(file_path):
    """Read CSV and auto-detect observed/predicted columns"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")

    print(f"\n FILE: {os.path.basename(file_path)}")
    print(f"Columns: {df.columns.tolist()}")

    obs_col = None
    pred_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["observe", "actual", "true", "observation", "real"]):
            obs_col = col
        if any(kw in col_lower for kw in ["predict", "pred", "prediction"]):
            pred_col = col

    if not obs_col or not pred_col:
        obs_col, pred_col = df.columns[0], df.columns[1]
        print(f"Warning: No keyword columns detected, using default: {obs_col}, {pred_col}")

    x_obs = df[obs_col].dropna().values.astype(np.float32)
    y_pred = df[pred_col].dropna().values.astype(np.float32)
    min_len = min(len(x_obs), len(y_pred))
    return x_obs[:min_len], y_pred[:min_len], obs_col, pred_col

# ------------------------------------------------------
# Core: Draw a single subplot
# ------------------------------------------------------
def plot_subplot(ax, x_obs, y_pred, file_name, obs_col, pred_col, marker, model_name, data_type):
    """Draw scatter plot with R² annotation"""
    r2 = r2_score(x_obs, y_pred)

    ax.scatter(
        x_obs, y_pred,
        color=COLOR,
        s=POINT_SIZE,
        alpha=ALPHA,
        edgecolors="none"
    )

    min_val = min(x_obs.min(), y_pred.min())
    max_val = max(x_obs.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1)

    if data_type == "DO":
        xlabel_text = f"Observed DO\n{marker}{model_name}"
    else:
        xlabel_text = f"Observed TN\n{marker}{model_name}"

    ax.set_xlabel(xlabel_text, fontsize=10)
    ax.set_ylabel("Predicted Value (Mg/L)", fontsize=10)

    ax.grid(True, linestyle="--", alpha=0.3)

    x_range = max_val - min_val
    y_range = max_val - min_val
    ax.set_xlim(min_val - 0.1 * x_range, max_val + 0.1 * x_range)
    ax.set_ylim(min_val - 0.1 * y_range, max_val + 0.1 * y_range)

    ax.text(
        0.05, 0.95,
        f"$R^2$={r2:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top'
    )

# ------------------------------------------------------
# Main: Generate 12 subplots (4 rows × 3 columns)
# ------------------------------------------------------
def main():
    fig, axes = plt.subplots(4, 3, figsize=FIGSIZE)
    axes_flat = axes.flatten()

    markers = []
    for i in range(12):
        marker_index = i % 6
        marker = f"({chr(97 + marker_index)}) "
        markers.append(marker)

    model_names = [
        "CNN", "LSTM", "BiLSTM", "CNN-OTDS", "LSTM-OTDS", "BiLSTM-OTDS",
        "CNN", "LSTM", "BiLSTM", "CNN-OTDS", "LSTM-OTDS", "BiLSTM-OTDS"
    ]
    data_types = ["DO"] * 6 + ["TN"] * 6

    for i, file_path in enumerate(FILE_PATHS):
        if i >= 12:
            break
        try:
            model_name = model_names[i]
            data_type = data_types[i]
            x_obs, y_pred, obs_col, pred_col = load_data(file_path)

            plot_subplot(
                ax=axes_flat[i],
                x_obs=x_obs,
                y_pred=y_pred,
                file_name=file_path,
                obs_col=obs_col,
                pred_col=pred_col,
                marker=markers[i],
                model_name=model_name,
                data_type=data_type
            )
        except Exception as e:
            print(f"Failed to process: {file_path}, Error: {str(e)}")
            axes_flat[i].text(0.5, 0.5, f"Error:\n{str(e)}",
                              ha="center", va="center", fontsize=10)
            axes_flat[i].set_title(f"File {i+1} (Error)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight")
    print(f"\nFigure saved to: {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()