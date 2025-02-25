# 학습과 평가를 동시에 진행하고 결과를 저장하는 스크립트
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
import json

# 사용자 정의 모델들
from models.vanilla_rnn import VanillaRNN
from models.vanilla_lstm import VanillaLSTM
from models.vanilla_gru import VanillaGRU

# 사용자 정의 함수들
from utils.functions import train_model, evaluate_model, get_csv_files

# 데이터셋
from datasetes.dataset_rnn import RNNTimeSeriesDataset
from preprocessing.data_prepro import abnormal_0, abnormal_1  # 리스트 임포트
from preprocessing.data_prepro import load_csv_with_issue_handling
from preprocessing.data_prepro import get_window_data_for_rnn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_type = args.model_type
    if model_type == 'rnn':
        parameters = {"model_type": 'rnn', "batch_size": 32, "lr": 0.001, "window_size": 10, "hidden_dim": 256}

    elif model_type == 'lstm':
        parameters = {"model_type": 'lstm', "batch_size": 16, "lr": 0.001, "window_size": 10, "hidden_dim": 256}

    elif model_type == 'gru':
        parameters = {"model_type": 'gru', "batch_size": 32, "lr": 0.001, "window_size": 20, "hidden_dim": 128}

    else:
        raise ValueError(f"Invalid model type: {args.model_type}. Choose from ['rnn', 'lstm', 'gru']")

    batch_size = parameters['batch_size']
    lr = parameters['lr']
    window_size = parameters['window_size']
    hidden_dim = parameters['hidden_dim']
    num_epochs = args.num_epoch
    groups = args.groups

    print(f"Model Type: {model_type}, batch_size: {batch_size}, lr: {lr}, window_size: {window_size}, hidden_dim: {hidden_dim}, num_epochs: {num_epochs}, group: {groups}")

    for run_idx in range(args.num_runs):

        if model_type == 'rnn':
            model = VanillaRNN(input_dim=20, hidden_dim=parameters['hidden_dim'], output_dim=4).to(device)

        elif model_type == 'lstm':
            model = VanillaLSTM(input_dim=20, hidden_dim=parameters['hidden_dim'], output_dim=4).to(device)

        elif model_type == 'gru':
            model = VanillaGRU(input_dim=20, hidden_dim=parameters['hidden_dim'], output_dim=4).to(device)

        else:
            raise ValueError(f"Invalid model type: {model_type}. Choose from ['rnn', 'lstm', 'gru']")

        print(f"Run {run_idx + 1}/{args.num_runs}")

        # train/test 파일 선정
        train_files, test_files = [], []
        for group in groups:
            group_path = os.path.join('../data/Group_modified', group)
            csv_files = get_csv_files(group_path)
            valid_files = [f for f in csv_files if os.path.basename(f) not in abnormal_0]
            valid_files_for_test = [f for f in valid_files if os.path.basename(f) not in abnormal_1]
            test_file = np.random.choice(valid_files_for_test, 1)[0]
            test_files.append({"group": group, "test_file": test_file})
            train_files.extend([f for f in valid_files if f != test_file])

        # tset 파일 정보 저장
        # run_dir: TensorBoard 로그 및 체크포인트 저장 경로
        run_dir = os.path.join("runs", model_type, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        test_file_path = os.path.join(run_dir, "test_files.json")
        with open(test_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_files, f, indent=4, ensure_ascii=False)

        # TensorBoard writer
        writer = SummaryWriter(run_dir)

        # abnormal 파일 처리 후 최종 train 파일 선정 후 concat
        train_data_X, train_data_y = [], []
        for f in train_files:
            df_train = load_csv_with_issue_handling(f)
            X_train, y_train = get_window_data_for_rnn(df_train, window_size, scale_data=False)
            train_data_X.append(X_train)
            train_data_y.append(y_train)

        X_train_all = np.concatenate(train_data_X, axis=0)
        y_train_all = np.concatenate(train_data_y, axis=0)

        # trian 데이터 스케일링 및 Dataset, DataLoader 생성
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        N, T, D = X_train_all.shape
        X_2d = X_train_all.reshape(N * T, D)
        scaler_X.fit(X_2d)
        scaler_y.fit(y_train_all)

        X_2d_scaled = scaler_X.transform(X_2d)
        X_train_scaled = X_2d_scaled.reshape(N, T, D)
        y_train_scaled = scaler_y.transform(y_train_all)

        train_dataset = RNNTimeSeriesDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # validation data
        val_data_X, val_data_y = [], []
        for test_entry in test_files:
            df_val = load_csv_with_issue_handling(test_entry['test_file'])
            X_val, y_val = get_window_data_for_rnn(df_val, window_size, scale_data=False)
            val_data_X.append(X_val)
            val_data_y.append(y_val)

        X_val_all = np.concatenate(val_data_X, axis=0)
        y_val_all = np.concatenate(val_data_y, axis=0)

        # validation 데이터 스케일링 및 Dataset, DataLoader 생성
        N_val, T_val, D_val = X_val_all.shape
        X_val_2d = X_val_all.reshape(N_val * T_val, D_val)
        X_val_2d_scaled = scaler_X.transform(X_val_2d)
        X_val_scaled = X_val_2d_scaled.reshape(N_val, T_val, D_val)
        y_val_scaled = scaler_y.transform(y_val_all)

        val_dataset = RNNTimeSeriesDataset(X_val_scaled, y_val_scaled)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss 및 Optimizer 설정
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        patience = 20
        counter = 0
        best_val_loss = float('inf')

        # Training
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch, writer=writer)
            val_loss, val_trues, val_preds = evaluate_model(model, val_loader, criterion, device, epoch, writer=writer, phase="Val")

            tqdm.write(f"Epoch {epoch}/{num_epochs} | "f"train={train_loss:.4f}, val={val_loss:.4f}")

            # best 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                save_path = os.path.join(checkpoints_dir, f"best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }, save_path)
                print(f"Best model so far val_loss={val_loss:.4f} with epoch={epoch}")
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter}/{patience} (Best val_loss={best_val_loss:.4f})")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for training a model.")
    parser.add_argument('--model_type', type=str, default='gru', help="Type of model to train")
    parser.add_argument('--groups', nargs='+', type=str,
                        default=['air gasification',
                                 'steam gasification',
                                 'CO2 gasification',
                                 'O2steam gasification'], help="Group to train (Fix)")
    parser.add_argument('--num_epoch', type=int, default=100, help="Number of epochs")
    parser.add_argument('--num_runs', type=int, default=10, help="Number of runs")
    args = parser.parse_args()

    main()