import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functools import partial
from config import *
from utils.data_loader import load_data, WaterQualityDataset
from models.bilstm import BiLSTMModel
from losses.sam_fgsm import sam_fgsm_loss
from utils.train_eval import run_experiment

# Data Loading
features, labels, scaler, mean, std, _ = load_data(TRAIN_PATH, is_train=True)
X_train_val, _, y_train_val, _ = train_test_split(features, labels, test_size=0.01, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)
train_dataset = WaterQualityDataset(X_train, y_train, SEQ_LENGTH)
val_dataset = WaterQualityDataset(X_val, y_val, SEQ_LENGTH)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, drop_last=True)
X_test, y_test, _, _, _, _ = load_data(TEST_PATH, scaler, mean, std, is_train=False)
test_dataset = WaterQualityDataset(X_test, y_test, SEQ_LENGTH)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=True)

# Model
model = BiLSTMModel(
    input_size=5,
    lstm_hidden_size=32,
    lstm_layers=1,
    lstm_dropout=0.2,
    fc_dropout=0.2,
    output_size=1
)

# Loss function (fixed hyperparameters)
loss_func = partial(sam_fgsm_loss,
                    epsilon_sam=0.001, lambda_sam=0.01,
                    epsilon_fgsm=0.001, lambda_fgsm=0.01)

# experiment
results = run_experiment(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    loss_func=loss_func,
    optimizer_config={'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
    device=DEVICE,
    epochs=EPOCHS,
    patience=PATIENCE,
    model_save_path=os.path.join(MODELS_DIR, "bilstm_sam_fgsm.pth")
)

import json
out_path = os.path.join(RESULTS_DIR, "bilstm_sam_fgsm.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"The result has been saved to {out_path}")