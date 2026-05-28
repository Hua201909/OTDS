Fitting Plot.py (English Version)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------------------
# Configuration parameters
# ------------------------------------------------------
FILE_PATHS = [
    "C:/Users/LEGION/Desktop/fsdownload/zhexiantu/output_Cnn_fgsm_sam_DO/cnn_fgsm_sam_curve_comparison.csv",  # DO
    "C:/Users/LEGION/Desktop/fsdownload/zhexiantu/output_LSTM_FGSM_SAM_DO/LSTM_FGSM_SAM_curve_comparison.csv",  # DO
    "C:/Users/LEGION/Desktop/fsdownload/zhexiantu/output_BiLSTM_fgsm-sam_DO/BiLSTM_fgsm-sam_curve_comparison.csv",  # DO
    "C:/Users/LEGION/Desktop/fsdownload/zhexiantu/output_CNN_FGSM_SAM_only_T/CNN_FGSM_SAM_curve_comparison.csv",  # TN
    "C:/Users/LEGION/Desktop/fsdownload/zhexiantu/output_LSTM_FGSM_SAM_T/LSTM_FGSM_curve_comparison.csv",  # TN
    "C:/Users/LEGION/Desktop/fsdownload/zhexiantu/output_BiLSTM_FGSM_SAM_T/BiLSTM_FGSM_SAM_curve_comparison.csv",  # TN
]
MODEL_NAMES = [
    "CNN-OTDS",
    "LSTM-OTDS",
    "BiLSTM-OTDS",
    "CNN-OTDS",
    "LSTM-OTDS",
    "BiLSTM-OTDS"
]
DATA_TYPES = ["DO", "DO", "DO", "TN", "TN", "TN"]
FIGSIZE = (20, 15)  # Figure size for 2 rows and 3 columns of subplots
DPI = 300  # Reduce resolution to avoid oversized file (adjustable up to 1000)
SAVE_PATH = "2x3_subplots_DO_TN.png"  # Image saving path
LINE_WIDTH = 1.5  # Polyline width
DO_Y_MIN = 6.0  # Minimum Y-axis value for DO subplots
TN_Y_MIN = 1.35  # Minimum Y-axis value for TN subplots
# ------------------------------------------------------
# Core Function: Automatically detect column names and draw subplots
# ------------------------------------------------------
def plot_single_subplot(ax, file_path, model_name, data_type, y_min):
    """Draw line charts of a single model on the specified subplot"""
    # Read CSV data (encoding adaptation)
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")
    # Automatically detect column names
    actual_col = None
    predict_col = None
    for col in df.columns:
        col_lower = col.lower()
        # Observed value column
        if any(kw in col_lower for kw in ["actual", "true", "observe"]):
            actual_col = col
        # Predicted value column
        if any(kw in col_lower for kw in ["predict", "pred"]) or model_name.replace("-", "") in col_lower:
            predict_col = col
    # Fallback strategy: use the first two columns by default
    if not actual_col:
        actual_col = df.columns[0]
    if not predict_col:
        predict_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    # Extract data
    actual = df[actual_col].dropna().values.astype(np.float32)
    predict = df[predict_col].dropna().values.astype(np.float32)
    min_len = min(len(actual), len(predict))
    actual = actual[:min_len]
    predict = predict[:min_len]
    # Draw polylines according to data type
    if data_type == "TN":
        ax.plot(range(min_len), actual, linestyle="--", color="blue", linewidth=LINE_WIDTH, label="Observed TN")
        ax.plot(range(min_len), predict, linestyle="-", color="orange", linewidth=LINE_WIDTH,
                label=f"Pred ({model_name})")
        ax.set_ylabel("TN (Mg/L)", fontsize=15)
        ax.set_ylim(bottom=y_min)
    else:  # DO data
        ax.plot(range(min_len), actual, linestyle="--", color="blue", linewidth=LINE_WIDTH, label="Observed DO")
        ax.plot(range(min_len), predict, linestyle="-", color="orange", linewidth=LINE_WIDTH,
                label=f"Pred ({model_name})")
        ax.set_ylabel("DO (Mg/L)", fontsize=15)
        ax.set_ylim(bottom=y_min)
    # --------------------------------------------------
    # Modification: Place all legends at the upper right corner
    # --------------------------------------------------
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
# ------------------------------------------------------
# Main Program: Draw 2×3 subplot layout
# ------------------------------------------------------
def plot_2x3_subplots(file_paths, model_names, data_types):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
    axes = axes.flatten()  # Flatten 2D array to 1D array
 for i, (file_path, model_name, data_type) in enumerate(zip(file_paths, model_names, data_types)):
        ax = axes[i]
        y_min = DO_Y_MIN if i < 3 else TN_Y_MIN
        plot_single_subplot(ax, file_path, model_name, data_type, y_min)
        ax.set_xlabel("Hourly Point Index", fontsize=15)
    # Adjust subplot spacing
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    # Save image file
    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches='tight')
    print(f"\nImage saved to: {SAVE_PATH}")
    plt.show()
# ------------------------------------------------------
# Execute plotting function
# ------------------------------------------------------
if __name__ == "__main__":
    plot_2x3_subplots(FILE_PATHS, MODEL_NAMES, DATA_TYPES)