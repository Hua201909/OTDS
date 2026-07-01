import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import matplotlib.pyplot as plt
import ssl
import matplotlib as mpl
import os
import time
from scipy.stats import spearmanr

# -------------------------- 1. Basic environment configuration --------------------------
mpl.rcParams.update({
    'font.family': ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 1000,
    'savefig.dpi': 1000,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

ssl._create_default_https_context = ssl._create_unverified_context

# -------------------------- 2. Data loading & Dataset definition --------------------------
def load_data(file_path, scaler_features=None, label_mean=None, label_std=None, is_train=True):
    """Load data: feature order strictly [T, pH, DO, EC, NTU]"""
    data = pd.read_excel(file_path).dropna()
    features_df = data[['T', 'PH', 'DO', 'EC', 'NTU']]
    labels_series = data['DO']

    features_np = features_df.values.astype(np.float32)
    labels_np = labels_series.values.astype(np.float32)

    if is_train:
        scaler_features = StandardScaler()
        features_scaled = scaler_features.fit_transform(features_np)
        label_mean = labels_np.mean()
        label_std = labels_np.std() if labels_np.std() != 0 else 1
    else:
        if scaler_features is None:
            raise ValueError("Test set must receive scaler from training set!")
        features_scaled = scaler_features.transform(features_np)

    return features_scaled, labels_np, scaler_features, label_mean, label_std, data

class DODataSet(Dataset):
    """Time-series dataset: generates [seq_length, 5] inputs and single-step labels"""
    def __init__(self, features, labels, seq_length=128):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.features[idx:idx + self.seq_length]
        y_target = self.labels[idx + self.seq_length]
        return torch.tensor(x_seq), torch.tensor(y_target)

# -------------------------- 3. LSTM model definition --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, lstm_hidden_size=64, lstm_layers=1, lstm_dropout=0.2, fc_dropout=0.2,
                 output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        return self.fc(self.fc_dropout(last_step_out))

# -------------------------- 4. Core: Permutation importance calculation --------------------------
def permutation_importance(model, data_loader, cond_feat_idx, plot_feat_names, device, n_repeats=10):
    """
    Permutation importance: RMSE performance drop (simulates SHAP Mean(|SHAP Value|))
    - cond_feat_idx: conditional feature indices (order from paper: Turbidity→Conductivity→Water Temp→pH)
    - plot_feat_names: X-axis labels (English)
    """
    model.eval()
    baseline_rmse = 0.0
    n_samples = 0

    # Step 1: baseline RMSE (original features)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            batch_mse = nn.MSELoss()(outputs, labels).item()
            baseline_rmse += np.sqrt(batch_mse) * inputs.size(0)
            n_samples += inputs.size(0)
    baseline_rmse /= n_samples
    print(f"Baseline RMSE (original features): {baseline_rmse:.6f}")

    # Step 2: importance for each feature
    importances = []
    errors = []
    for idx, feat_idx in enumerate(cond_feat_idx):
        feat_rmses = []
        for _ in range(n_repeats):
            perturbed_rmse = 0.0
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs_perturbed = inputs.clone()
                    for i in range(inputs_perturbed.shape[0]):
                        inputs_perturbed[i, :, feat_idx] = inputs_perturbed[
                            i, torch.randperm(inputs_perturbed.shape[1]), feat_idx]
                    outputs = model(inputs_perturbed).squeeze()
                    batch_mse = nn.MSELoss()(outputs, labels).item()
                    perturbed_rmse += np.sqrt(batch_mse) * inputs.size(0)
            perturbed_rmse /= n_samples
            feat_rmses.append(perturbed_rmse)

        feat_importance = np.mean(feat_rmses) - baseline_rmse
        importances.append(feat_importance)
        errors.append(np.std(feat_rmses))
        print(f"Feature {feat_idx} ({plot_feat_names[idx]}) importance: {feat_importance:.6f}")

    return np.array(importances), np.array(errors)

# -------------------------- 5. Helper: load pre-trained models --------------------------
def load_trained_models(config, device):
    """Load two models: with and without regularization"""
    model_with = LSTMModel(
        input_size=config['input_size'],
        lstm_hidden_size=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        lstm_dropout=config['dropout'],
        fc_dropout=config['dropout']
    ).to(device)

    model_without = LSTMModel(
        input_size=config['input_size'],
        lstm_hidden_size=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        lstm_dropout=config['dropout'],
        fc_dropout=config['dropout']
    ).to(device)

    try:
        model_with.load_state_dict(torch.load(config['weight_path_with'], map_location=device))
        model_without.load_state_dict(torch.load(config['weight_path_without'], map_location=device))
        model_with.eval()
        model_without.eval()
        print(f"\n✅ Successfully loaded models:")
        print(f"  - With {config['regularization_type']}: {config['weight_path_with']}")
        print(f"  - Without {config['regularization_type']}: {config['weight_path_without']}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"\n❌ Weight files not found! Please check paths:\n"
            f"1. With {config['regularization_type']}: {config['weight_path_with']}\n"
            f"2. Without {config['regularization_type']}: {config['weight_path_without']}"
        ) from e

    return model_with, model_without

# -------------------------- 6. Helper: prepare data (full dataset, no sample limit) --------------------------
def prepare_data(config):
    """Prepare test data using full dataset (no truncation)"""
    features_train, labels_train, scaler, _, _, _ = load_data(config['file_path_train'], is_train=True)
    features_test, labels_test, _, _, _, _ = load_data(
        config['file_path_test'],
        scaler_features=scaler,
        is_train=False
    )

    test_dataset = DODataSet(features_test, labels_test, config['seq_length'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True
    )

    # Feature mapping for paper figures: [Turbidity, Conductivity, Water Temp, pH]
    feat_mapping = {
        'raw_feats_full': ['Water Temp (℃)', 'pH (dimensionless)', 'DO (mg/L)', 'Conductivity (μs/cm)', 'Turbidity (NTU)'],
        'cond_feat_idx': [4, 3, 0, 1],
        'plot_feat_names': ['Turbidity', 'Conductivity', 'Water Temp', 'pH']
    }

    return {
        'test_loader': test_loader,
        'test_dataset_size': len(test_dataset),
        'feat_mapping': feat_mapping,
        'scaler': scaler
    }

# -------------------------- 7. Core visualization: 2×2 subplots --------------------------
def plot_shap_comparison(imp_with, err_with, imp_without, err_without, feat_mapping, config, device, test_loader,
                         model_with, model_without):
    """
    Generate 2×2 subplots identical to paper figures:
    Top-left: importance comparison | Top-right: importance distribution
    Bottom-left: rank change | Bottom-right: importance change
    All labels in English, legend moved to upper right.
    """
    plot_feat_names = feat_mapping['plot_feat_names']
    cond_feat_idx = feat_mapping['cond_feat_idx']
    regularization_type = config['regularization_type']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        f'SHAP Analysis: Impact of {regularization_type} on Feature Importance (Condition Features Only)',
        fontsize=12, y=1.02
    )

    # Top-left: Importance comparison
    x = np.arange(len(plot_feat_names))
    width = 0.35
    bars1 = axes[0, 0].bar(
        x - width/2, imp_with, width,
        label=f'With {regularization_type}',
        color='#1f77b4', edgecolor='black', linewidth=0.8,
        yerr=err_with, capsize=3
    )
    bars2 = axes[0, 0].bar(
        x + width/2, imp_without, width,
        label=f'Without {regularization_type}',
        color='#ff7f0e', edgecolor='black', linewidth=0.8,
        yerr=err_without, capsize=3
    )
    axes[0, 0].set_xlabel('Features')
    axes[0, 0].set_ylabel('Importance (RMSE Performance Drop)')
    axes[0, 0].set_title('Condition Features Importance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(plot_feat_names)
    axes[0, 0].legend(loc='upper right', frameon=True)
    axes[0, 0].grid(alpha=0.3, linestyle='--', axis='x')

    # Top-right: Importance distribution
    def get_importance_dist(model, feat_idx, device):
        model.eval()
        baseline_rmse = 0.0
        n_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                batch_mse = nn.MSELoss()(outputs, labels).item()
                baseline_rmse += np.sqrt(batch_mse) * inputs.size(0)
                n_samples += inputs.size(0)
        baseline_rmse /= n_samples

        dists = []
        for _ in range(20):
            perturbed_rmse = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs_perturbed = inputs.clone()
                    for i in range(inputs_perturbed.shape[0]):
                        inputs_perturbed[i, :, feat_idx] = inputs_perturbed[
                            i, torch.randperm(inputs_perturbed.shape[1]), feat_idx]
                    inputs_perturbed, labels = inputs_perturbed.to(device), labels.to(device)
                    outputs = model(inputs_perturbed).squeeze()
                    batch_mse = nn.MSELoss()(outputs, labels).item()
                    perturbed_rmse += np.sqrt(batch_mse) * inputs.size(0)
            dists.append((perturbed_rmse / n_samples) - baseline_rmse)
        return np.array(dists)

    dist_with = np.concatenate([get_importance_dist(model_with, idx, device) for idx in cond_feat_idx])
    dist_without = np.concatenate([get_importance_dist(model_without, idx, device) for idx in cond_feat_idx])

    bins = np.linspace(min(dist_with.min(), dist_without.min()), max(dist_with.max(), dist_without.max()), 30)
    axes[0, 1].hist(dist_with, bins=bins, label=f'With {regularization_type}', color='#1f77b4', alpha=0.7)
    axes[0, 1].hist(dist_without, bins=bins, label=f'Without {regularization_type}', color='#ff7f0e', alpha=0.7)
    axes[0, 1].set_xlabel('Feature Importance (RMSE Performance Drop)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Feature Importance Distribution')
    axes[0, 1].legend(loc='upper right', frameon=True)
    axes[0, 1].grid(alpha=0.3, linestyle='--', axis='x')

    # Bottom-left: Rank change
    rank_with = np.argsort(np.argsort(-imp_with)) + 1
    rank_without = np.argsort(np.argsort(-imp_without)) + 1
    corr, p_val = spearmanr(rank_with, rank_without)
    corr_rounded = round(corr, 3)

    for i, feat in enumerate(plot_feat_names):
        axes[1, 0].scatter(
            rank_with[i], rank_without[i],
            s=60, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.8
        )
        axes[1, 0].text(rank_with[i] + 0.05, rank_without[i], feat, fontsize=8, ha='left', va='center')

    line_x = [min(rank_with), max(rank_with)]
    line_y = [min(rank_without), max(rank_without)] if corr_rounded > 0 else [max(rank_without), min(rank_without)]
    axes[1, 0].plot(
        line_x, line_y, 'r--', linewidth=1.5, alpha=0.7,
        label=f'Perfect Correlation (r={corr_rounded})'
    )
    axes[1, 0].set_xlabel(f'Rank (With {regularization_type})')
    axes[1, 0].set_ylabel(f'Rank (Without {regularization_type})')
    axes[1, 0].set_title(f'Rank Change (Correlation: {corr_rounded})')
    axes[1, 0].legend(loc='upper left', frameon=True)
    axes[1, 0].grid(alpha=0.3, linestyle='--')
    axes[1, 0].set_xlim(0.8, 4.2)
    axes[1, 0].set_ylim(0.8, 4.2)

    # Bottom-right: Impact of regularization on condition features
    imp_change = imp_with - imp_without
    colors = ['#2ca02c' if delta > 0 else '#d62728' for delta in imp_change]
    bars = axes[1, 1].barh(plot_feat_names, imp_change, color=colors, edgecolor='black', linewidth=0.8)
    axes[1, 1].set_xlabel(f'Importance Change (With - Without {regularization_type})')
    axes[1, 1].set_title(f'Impact of {regularization_type} on Condition Features')
    axes[1, 1].grid(alpha=0.3, linestyle='--', axis='x')
    axes[1, 1].axvline(x=0, color='black', linewidth=1.0, alpha=0.7)

    # Save PDF (vector format, DPI=300)
    os.makedirs(config['output_dir'], exist_ok=True)
    pdf_name = f'shap_{config["model_type"]}_{regularization_type}_comparison.pdf'
    pdf_path = os.path.join(config['output_dir'], pdf_name)
    fig.savefig(
        pdf_path,
        dpi=1000,
        bbox_inches='tight',
        format='pdf'
    )
    plt.close()
    print(f"\n🎉 Figure saved to: {pdf_path} (DPI=300)")

    return rank_with, rank_without, corr_rounded

# -------------------------- 8. Save results (CSV + summary) --------------------------
def save_results(imp_with, err_with, imp_without, err_without, rank_with, rank_without, corr, feat_mapping, config):
    """Save CSV table and statistical summary (all in English)"""
    plot_feat_names = feat_mapping['plot_feat_names']
    regularization_type = config['regularization_type']

    df_results = pd.DataFrame({
        'Feature': plot_feat_names,
        f'Importance_With_{regularization_type}': imp_with,
        f'Std_With_{regularization_type}': err_with,
        f'Importance_Without_{regularization_type}': imp_without,
        f'Std_Without_{regularization_type}': err_without,
        f'Importance_Change': imp_with - imp_without,
        f'Rank_With_{regularization_type}': rank_with,
        f'Rank_Without_{regularization_type}': rank_without
    })
    df_results = df_results.sort_values(f'Importance_With_{regularization_type}', ascending=False)

    csv_path = os.path.join(config['output_dir'], f'{regularization_type}_shap_results.csv')
    df_results.to_csv(csv_path, index=False, encoding='utf-8', float_format='%.6f')

    summary_path = os.path.join(config['output_dir'], f'{regularization_type}_shap_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{config['model_type']} Model Feature Importance (SHAP) Analysis Summary\n")
        f.write(f"Regularization Type: {regularization_type}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Sample Configuration\n")
        f.write(f"   - Data Usage: Full dataset (no truncation)\n")
        f.write(f"   - Sequence Length: {config['seq_length']}\n\n")

        f.write("2. Feature Mapping (consistent with paper figures)\n")
        f.write(f"   Raw Features (with indices): {feat_mapping['raw_feats_full']}\n")
        f.write(f"   Paper Figure Feature Order (indices): {feat_mapping['cond_feat_idx']} → {plot_feat_names}\n\n")

        f.write("3. Core Results\n")
        f.write(f"   - Feature Rank Correlation (Spearman): {corr:.3f}\n")
        f.write(f"   - Top 3 Important Features (With {regularization_type}):\n")
        for i, (_, row) in enumerate(df_results.head(3).iterrows()):
            f.write(
                f"     {i+1}. {row['Feature']}: {row[f'Importance_With_{regularization_type}']:.6f} (Rank: {row[f'Rank_With_{regularization_type}']})\n")

        f.write(f"\n4. Impact of {regularization_type}\n")
        top_change = df_results.nlargest(2, 'Importance_Change')
        bottom_change = df_results.nsmallest(2, 'Importance_Change')
        f.write("   Features with Most Increased Importance:\n")
        for _, row in top_change.iterrows():
            f.write(f"     - {row['Feature']}: +{row['Importance_Change']:.6f}\n")
        f.write("   Features with Most Decreased Importance:\n")
        for _, row in bottom_change.iterrows():
            f.write(f"     - {row['Feature']}: {row['Importance_Change']:.6f}\n")

    print(f"\n📊 Results saved:")
    print(f"   - CSV: {csv_path}")
    print(f"   - Summary: {summary_path}")

    return df_results

# -------------------------- 9. Main function --------------------------
def main():
    print("=" * 80)
    print(f"LSTM Model Feature Importance (SHAP) Analysis Program (Full Dataset, DPI=300)")
    print("=" * 80)

    # -------------------------- Configuration (switch between Sinkhorn/FGSM+SAM) --------------------------
    config = {
        # Data paths
        'file_path_train': 'C:/Users/LEGION/Desktop/fsdownload/Training Data/Guanyan Training Data 2021.1-2022.5.xlsx',
        'file_path_test': 'C:/Users/LEGION/Desktop/fsdownload/Prediction Data/Guanyan Precise Data 2022.6-2022.10 (Use last 15 days for precise prediction).xlsx',

        # Model & regularization
        'model_type': 'LSTM',
        'regularization_type': 'OTDS',   # Options: 'FGSM+SAM' or 'Sinkhorn'
        'weight_path_with': 'C:/Users/LEGION/Desktop/fsdownload/best_model_lstm_sam_fgsm.pth',
        'weight_path_without': 'C:/Users/LEGION/Desktop/fsdownload/best_lstm_no_cnn_no_sam.pth',

        # Time-series parameters
        'seq_length': 128,
        'batch_size': 64,

        # Model architecture
        'input_size': 5,
        'lstm_hidden': 64,
        'lstm_layers': 1,
        'dropout': 0.2,

        # Output directory
        'output_dir': 'C:/Users/LEGION/Desktop/shap',
    }

    # Step 1: Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    print(f"\n1. Device: {device}")

    # Step 2: Data
    print(f"\n2. Preparing data (full dataset)...")
    try:
        data_dict = prepare_data(config)
        test_loader = data_dict['test_loader']
        feat_mapping = data_dict['feat_mapping']
        print(f"   - Test set time-series samples: {data_dict['test_dataset_size']} (full dataset)")
        print(f"   - Feature mapping: {feat_mapping['plot_feat_names']}")
    except Exception as e:
        print(f"   ❌ Data preparation failed: {e}")
        return

    # Step 3: Load models
    print(f"\n3. Loading pre-trained models ({config['regularization_type']} comparison)...")
    try:
        model_with, model_without = load_trained_models(config, device)
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return

    # Step 4: Calculate importance
    print(f"\n4. Calculating permutation importance (RMSE metric)...")
    try:
        print(f"\n   === Model With {config['regularization_type']} ===")
        imp_with, err_with = permutation_importance(
            model=model_with,
            data_loader=test_loader,
            cond_feat_idx=feat_mapping['cond_feat_idx'],
            plot_feat_names=feat_mapping['plot_feat_names'],
            device=device,
            n_repeats=10
        )

        print(f"\n   === Model Without {config['regularization_type']} ===")
        imp_without, err_without = permutation_importance(
            model=model_without,
            data_loader=test_loader,
            cond_feat_idx=feat_mapping['cond_feat_idx'],
            plot_feat_names=feat_mapping['plot_feat_names'],
            device=device,
            n_repeats=10
        )
    except Exception as e:
        print(f"   ❌ Importance calculation failed: {e}")
        return

    # Step 5: Generate figures
    print(f"\n5. Generating 2×2 subplots (paper format, DPI=1000)...")
    try:
        rank_with, rank_without, corr = plot_shap_comparison(
            imp_with=imp_with,
            err_with=err_with,
            imp_without=imp_without,
            err_without=err_without,
            feat_mapping=feat_mapping,
            config=config,
            device=device,
            test_loader=test_loader,
            model_with=model_with,
            model_without=model_without
        )
    except Exception as e:
        print(f"   ❌ Figure generation failed: {e}")
        return

    # Step 6: Save results
    print(f"\n6. Saving analysis results...")
    try:
        df_results = save_results(
            imp_with=imp_with,
            err_with=err_with,
            imp_without=imp_without,
            err_without=err_without,
            rank_with=rank_with,
            rank_without=rank_without,
            corr=corr,
            feat_mapping=feat_mapping,
            config=config
        )
    except Exception as e:
        print(f"   ❌ Result saving failed: {e}")
        return

    # Step 7: Print key findings
    print("\n" + "=" * 80)
    print(f"Key Findings ({config['regularization_type']} Comparison, Full Dataset)")
    print("=" * 80)
    print(f"\n1. Feature Rank Correlation: Spearman r = {corr:.3f}")
    print(f"   - If r = -1.000: {config['regularization_type']} completely reversed ranks")
    print(f"   - If r = +1.000: {config['regularization_type']} did not change ranks")
    print(f"\n2. Top features (With {config['regularization_type']}):")
    for i, (_, row) in enumerate(df_results.iterrows()):
        col_name = f'Importance_With_{config["regularization_type"]}'
        print(f"  {i+1}. {row['Feature']}: Importance={row[col_name]:.6f}")

    print("\n" + "=" * 80)
    print(f"Program completed. All results saved to: {config['output_dir']} (Figure DPI=300)")

if __name__ == "__main__":
    main()