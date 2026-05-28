OTDS: Optimal transport-inspired dual-space learning for robust spatiotemporal regression

This project compares three neural network architectures (LSTM, Bidirectional LSTM, CNN) and four training strategies (standard MSE, FGSM input perturbation, SAM parameter perturbation, SAM+FGSM combined) on water quality total nitrogen prediction. All experiments adopt identical training/test data, hyperparameters, and evaluation metrics to ensure fair comparison.

1. Project Overview
Objectives
- Evaluate the time-series modeling capability of different recurrent and convolutional neural networks for water quality data.
- Explore the impact of FGSM(Fast Gradient Sign Method) and SAM (Sharpness-Aware Minimization) adversarial training on prediction accuracy and robustness.
- Provide reproducible benchmark results for future research reference.

Data Description
- Features: Water Temperature (°C), pH Value, Electrical Conductivity (μs/cm), Turbidity (NTU), Total Nitrogen (mg/L).
- Label: Total Nitrogen (mg/L).
- Source: Training set `train.xlsx` (Jan 2021 – May 2022), Test set `test.xlsx` (Jun 2022 – Oct 2022).
- Preprocessing: Z-Score standardization (mean = 0, standard deviation = 1). Training set statistics are applied to the test set.
Model \&amp; Strategy Combinations \(12 in Total\)
Model Architecture	Training Strategy	Loss Function Description
LSTM	Baseline	Standard MSE Loss
LSTM	FGSM	MSE + λ·FGSM Adversarial Loss \(Input Perturbation\)

LSTM	SAM	MSE + λ·SAM Parameter Perturbation Loss
LSTM	SAM+FGSM	MSE + λ_sam·SAM + λ_fgsm·FGSM
双向LSTM	Baseline	Same as above
双向LSTM	FGSM	Same as above
双向LSTM	SAM	Same as above
双向LSTM	SAM+FGSM	Same as above
CNN	Baseline	Same as above
CNN	FGSM	Same as above
CNN	SAM	Same as above
CNN	SAM+FGSM	Same as above
All experiments share the same hyperparameters: sequence length 128, batch size 64, learning rate 0.005, AdamW optimizer, patience 20 for early stopping.

2. Environment Requirements
- Python 3.11+
- Dependencies(see `requirements.txt`):
pandas==1.5.3
numpy==1.24.3
torch==2.0.1
scikit-learn==1.3.0
matplotlib==3.7.1
openpyxl==3.1.2
tqdm==4.65.0

3. Quick Start
Install Dependencies
pip install -r requirements.txt
Prepare Data
Place your Excel files into the `data` folder and rename them as:
- Training data: `train.xlsx`
- Test data: `test.xlsx`
Configure Settings (Optional)
Edit `config.py` to adjust data paths, hyperparameters, or device configuration (GPU by default, auto fallback to CPU if unavailable).
Run All Experiments
python scripts/run_all.py
This script executes 12 experiments sequentially. Each model is trained, validated, and tested independently. All model checkpoints and outputs are saved to the `results` folder.
Run Single Experiment
python scripts/run_lstm_baseline.py

Naming rule for other scripts: `run_{model}_{strategy}.py`
`{model}` ∈ {lstm, bilstm, cnn}
`{strategy}` ∈ {baseline, fgsm, sam, sam_fgsm}

4. Result Explanation

Output File Structure
results
├── lstm_baseline.json
├── lstm_fgsm.json
├── ...
├── bilstm_sam_fgsm.json
├── cnn_sam.json
├── metrics_summary.csv           # Summary table of all experiments
└── figures/                      # Optional: comparison bar charts, prediction curves

Each `{model}_{strategy}.json` file contains:

json
{
  "test_metrics": {
    "MSE": 0.0182,
    "RMSE": 0.135,
    "MAE": 0.091,
    "SMAPE": 11.8
  },
  "best_model_path": "best_lstm_baseline.pth",
  "predictions_scaled": [...],
  "actuals_scaled": [...]
}

metrics_summary.csv` aggregates all experimental results for direct comparison\.

Metric Interpretation
- MSE (Mean Squared Error): Average squared error, sensitive to outliers.
- RMSE (Root Mean Squared Error): Square root of MSE, consistent with the original label unit (mg/L).
- MAE (Mean Absolute Error): Average absolute error, robust to outliers.
- SMAPE (Symmetric Mean Absolute Percentage Error): Symmetric percentage error, reflecting relative error (%).
In general, lower RMSE and SMAPE indicate higher prediction accuracy.

Organize files following the directory structure and package as `water\_quality\_benchmark\.zip`:
water_quality_benchmark
├── README.md
├── requirements.txt
├── config.py
├── models
│   ├── __init__.py
│   ├── lstm.py
│   ├── bilstm.py
│   └── cnn.py
├── losses
│   ├── __init__.py
│   ├── baseline.py
│   ├── fgsm.py
│   ├── sam.py
│   └── sam_fgsm.py
├── utils
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── train_eval.py
├── scripts
│   ├── run_all.py
│   ├── run_lstm_baseline.py
│   ├── run_lstm_fgsm.py
│   ├── run_lstm_sam.py
│   ├── run_lstm_sam_fgsm.py
│   ├── run_bilstm_baseline.py
│   ├── run_bilstm_fgsm.py
│   ├── run_bilstm_sam.py
│   ├── run_bilstm_sam_fgsm.py
│   ├── run_cnn_baseline.py
│   ├── run_cnn_fgsm.py
│   ├── run_cnn_sam.py
│   └── run_cnn_sam_fgsm.py
├── data                     # Empty folder, put train.xlsx and test.xlsx manually
└── results                  # Empty folder, auto-generated after running
