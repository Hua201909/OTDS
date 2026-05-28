import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import os

# ------------------------------------------------------
# Configuration parameters (please modify according to actual circumstances)
# ------------------------------------------------------
FILE_PATHS = [  # 12 CSV file paths (replace with your actual paths)
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

FIGSIZE = (12, 8)  #
DPI = 300  # Save the resolution of the image
SAVE_PATH = "twelve_subplots_result.png"  # Image saving path
POINT_SIZE = 5  # Size of scatter plot points
ALPHA = 0.6  # Scatter plot transparency
COLOR = "blue"  # Scatter plot color


# ------------------------------------------------------
# Core function: Read the file and extract data (automatically detect column names)
# ------------------------------------------------------
def load_data(file_path):
    """Read the CSV file and automatically detect the columns of observed values and predicted values"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")  # Try UTF-8 encoding
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")  # When failing, try using GBK encoding.

    # Print column names (for debugging purposes)
    print(f"\n FILE: {os.path.basename(file_path)}")
    print(f"column: {df.columns.tolist()}")

    # Automatically detect column names (keyword matching)
    obs_col = None
    pred_col = None
    for col in df.columns:
        col_lower = col.lower()
        # Observation value column (containing keywords such as "observe", "actual", "true", etc.)
        if any(kw in col_lower for kw in ["observe", "actual", "true", "观测", "真实"]):
            obs_col = col
        # Prediction value column (containing keywords such as predict, pred, and forecast)
        if any(kw in col_lower for kw in ["predict", "pred", "预测"]):
            pred_col = col

    # If no detection is made, use the first two columns (the fallback plan)
    if not obs_col or not pred_col:
        obs_col, pred_col = df.columns[0], df.columns[1]
        print(f"Warning: No keyword column was detected. Using default column instead: {obs_col}, {pred_col}")

    # Extract data (convert to numpy array)
    x_obs = df[obs_col].dropna().values.astype(np.float32)  # 观测值（去空值）
    y_pred = df[pred_col].dropna().values.astype(np.float32)  # 预测值（去空值）

    # Ensure that the data lengths are consistent (truncate the longer array)
    min_len = min(len(x_obs), len(y_pred))
    return x_obs[:min_len], y_pred[:min_len], obs_col, pred_col


# ------------------------------------------------------
# Core function: Draw a single subplot (with options for adding markers and model names)
# ------------------------------------------------------
def plot_subplot(ax, x_obs, y_pred, file_name, obs_col, pred_col, marker, model_name, data_type):
    """Draw a scatter plot on the subgraph ax and label the R² value"""
    # Calculate R²
    r2 = r2_score(x_obs, y_pred)

    # Draw a scatter plot
    ax.scatter(
        x_obs, y_pred,
        color=COLOR,
        s=POINT_SIZE,  # dot size
        alpha=ALPHA,  # transparency
        edgecolors="none"  # No edge lines
    )

    # Add a perfect prediction reference line (y = x)
    min_val = min(x_obs.min(), y_pred.min())
    max_val = max(x_obs.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1)  # 红色虚线

    # Set the axis labels (use different labels according to the data type)
    if data_type == "DO":
        xlabel_text = f"Guanyan:Observe value(DO)\n{marker}{model_name}"
    else:  # TN
        xlabel_text = f"Guanyan:Observe value(TN)\n{marker}{model_name}"

    ax.set_xlabel(xlabel_text, fontsize=10)
    ax.set_ylabel(f"Predict value(Mg/L)", fontsize=10)

    # Add grid lines (light dashed lines)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set the axis range (with a 10% margin) and grid lines (light dashed lines)
    x_range = max_val - min_val
    y_range = max_val - min_val
    ax.set_xlim(min_val - 0.1 * x_range, max_val + 0.1 * x_range)
    ax.set_ylim(min_val - 0.1 * y_range, max_val + 0.1 * y_range)

    # Move R² to the top left corner (without background frame)
    ax.text(
        0.05, 0.95,  # Axis coordinate system position (upper left corner)
        f"$R^2$={r2:.4f}",  # R² Text
        transform=ax.transAxes,  # Use the axis coordinate system (with a range of 0-1)
        fontsize=10,
        verticalalignment='top'  # Top alignment
    )


# ------------------------------------------------------
# Main program: Generate twelve subgraphs (the first 6 DO and the last 6 TN)
# ------------------------------------------------------
def main():
    # Create a 4-row by 3-column subplot layout (suitable for 12 subplots)
    fig, axes = plt.subplots(4, 3, figsize=FIGSIZE)

    # Flatten the axes array (for easier looping)
    axes_flat = axes.flatten()

    # Dynamic generation of tags (cycled among a-f)
    markers = []
    for i in range(12):
        # Calculate the tag index (0-5 cycle)
        marker_index = i % 6
        # Generate labels (a) - (f)
        marker = f"({chr(97 + marker_index)}) "
        markers.append(marker)

    # List of model names (first 6 correspond to DO, last 6 correspond to TN)
    model_names = [
        "CNN", "LSTM", "BiLSTM", "CNN-OTDS", "LSTM-OTDS", "BiLSTM-OTDS",  # DO模型
        "CNN", "LSTM", "BiLSTM", "CNN-OTDS", "LSTM-OTDS", "BiLSTM-OTDS"  # TN模型
    ]

    # Model name list (the first 6 correspond to DO, the last 6 correspond to TN) Data type list (the first 6 DO, the last 6 TN)
    data_types = ["DO"] * 6 + ["TN"] * 6

    # Process 12 files in a loop
    for i, file_path in enumerate(FILE_PATHS):
        if i >= 12:  # Make sure to process only 12 files.
            break

        try:
            # Obtain the model name and data type
            model_name = model_names[i]
            data_type = data_types[i]

            #  Load Data
            x_obs, y_pred, obs_col, pred_col = load_data(file_path)

            # Draw subplots (passing dynamically generated markers, model names and data types)
            plot_subplot(
                ax=axes_flat[i],
                x_obs=x_obs,
                y_pred=y_pred,
                file_name=file_path,
                obs_col=obs_col,
                pred_col=pred_col,
                marker=markers[i],  # Dynamic labeling (a)-(f) loop
                model_name=model_name,  # Dynamic Model Name
                data_type=data_type  # Data Type (DO/TN)
            )


        except Exception as e:

            print(f"Failed to process file: {file_path}, Error: {str(e)}")

            axes_flat[i].text(0.5, 0.5, f"Error:\n{str(e)}",

                              ha="center", va="center", fontsize=10)

            axes_flat[i].set_title(f"File {i + 1} (Error)")

        # Adjust subplot spacing to prevent label overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for main title
    plt.subplots_adjust(wspace=0.3, hspace=0.6)  # Width/height spacing between subplots

    # Save the figure with high resolution
    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight")
    print(f"\nFigure saved to: {SAVE_PATH}")

    # Display the figure
    plt.show()


if __name__ == "__main__":
    main()