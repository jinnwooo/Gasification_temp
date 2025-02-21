# 학습과 평가를 동시에 진행하고 결과를 저장하는 스크립트
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler

# 사용자 정의 모델들
from models.vanilla_rnn import VanillaRNN
from models.vanilla_lstm import VanillaLSTM
from models.vanilla_gru import VanillaGRU

# 사용자 정의 메트릭들
from utils.metrics import mean_absolute_percentage_error
from utils.metrics import symmetric_mean_absolute_percentage_error

# 사용자 정의 함수들
from utils.functions import train_model, evaluate_model, get_csv_files, calculate_shap_values, determine_sample_size

# 데이터셋
from datasetes.dataset_rnn import RNNTimeSeriesDataset
from preprocessing.data_prepro import abnormal_0, abnormal_1  # 리스트 임포트
from preprocessing.data_prepro import load_csv_with_issue_handling
from preprocessing.data_prepro import get_window_data_for_rnn

# 평가 지표
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

    summary_results_dir = os.path.join(f"../Results/{model_type}")
    os.makedirs(summary_results_dir, exist_ok=True)
    summary_results_path = os.path.join(summary_results_dir, "summary_results.csv")
    if not os.path.exists(summary_results_path):
        with open(summary_results_path, 'w') as f:
            f.write("Test_File,Target,MSE,MAE,R2,MAPE,SMAPE\n")

    shap_values_list = {target: [] for target in ["CO", "CO2", "CH4", "H2"]}


    for run_idx in range(args.num_runs):

        if model_type == 'rnn':
            model = VanillaRNN(input_dim=20, hidden_dim=parameters['hidden_dim'], output_dim=4).to(device)

        elif model_type == 'lstm':
            model = VanillaLSTM(input_dim=20, hidden_dim=parameters['hidden_dim'], output_dim=4).to(device)

        elif model_type == 'gru':
            model = VanillaGRU(input_dim=20, hidden_dim=parameters['hidden_dim'], output_dim=4).to(device)

        print(f"Run {run_idx + 1}/{args.num_runs}")

        # train/test 파일 선정
        train_files, test_files = [], []
        for group in groups:
            group_path = os.path.join('../data/Group_modified', group)
            csv_files = get_csv_files(group_path)
            valid_files = [f for f in csv_files if os.path.basename(f) not in abnormal_0]
            valid_files_for_test = [f for f in valid_files if os.path.basename(f) not in abnormal_1]
            test_file = np.random.choice(valid_files_for_test, 1)[0]
            test_files.append((group, test_file))
            train_files.extend([f for f in valid_files if f != test_file])

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

        train_data_size = len(train_loader.dataset)
        print(f"Train 데이터 전체 샘플 수: {train_data_size}")

        # validation data
        val_data_X, val_data_y = [], []
        for _, test_file in test_files:
            df_val = load_csv_with_issue_handling(test_file)
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

        # run_dir: TensorBoard 로그 및 체크포인트 저장 경로
        run_dir = os.path.join("runs", model_type, f"run_{run_idx}") # 수정 필요
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # TensorBoard writer
        writer = SummaryWriter(run_dir)

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
                print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # SHAP
        sample_size = determine_sample_size(train_loader)
        print(f"SHAP 샘플링 개수: {sample_size}")
        shap_values, X_sample = calculate_shap_values(model, train_loader, device)

        # SHAP 결과 저장 디렉토리 생성
        shap_results_dir = os.path.join(f"../SHAP", model_type, "Results", f"run_{run_idx}")
        os.makedirs(shap_results_dir, exist_ok=True)

        # 타겟별 SHAP 값 저장 (Numpy 파일)
        for i, target in enumerate(["CO", "CO2", "CH4", "H2"]):
            shap_values_target = shap_values[:, :, :, i]  # 특정 타겟에 대한 SHAP 값 추출
            np.save(os.path.join(shap_results_dir, f"shap_values_train_{target}_{run_idx}.npy"), shap_values_target)
            shap_values_list[target].append(shap_values_target)


        # best 모델 불러오기
        best_model_path = os.path.join(checkpoints_dir, f"best.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        scaler_X = checkpoint['scaler_X']
        scaler_y = checkpoint['scaler_y']

        # test data evaluate
        for group_name, test_file in test_files:
            df_test = load_csv_with_issue_handling(test_file)
            X_test, y_test = get_window_data_for_rnn(df_test, window_size, scale_data=False)

            N_test, T_test, D_test = X_test.shape
            X_test_2d = X_test.reshape(N_test * T_test, D_test)
            X_test_2d_scaled = scaler_X.transform(X_test_2d)
            X_test_scaled = X_test_2d_scaled.reshape(N_test, T_test, D_test)
            y_test_scaled = scaler_y.transform(y_test)

            test_dataset = RNNTimeSeriesDataset(X_test_scaled, y_test_scaled)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            test_loss, test_trues, test_preds = evaluate_model(model, test_loader, criterion, device, epoch, writer=writer, phase="Test")

            test_trues_orig = scaler_y.inverse_transform(test_trues)
            test_preds_orig = scaler_y.inverse_transform(test_preds)

            # 성능 지표 계산
            mae_test = mean_absolute_error(test_trues_orig, test_preds_orig, multioutput='raw_values').tolist()
            mse_test = mean_squared_error(test_trues_orig, test_preds_orig, multioutput='raw_values').tolist()
            r2_test = [r2_score(test_trues_orig[:, i], test_preds_orig[:, i]) for i in range(test_trues_orig.shape[1])]
            mape_test = [mean_absolute_percentage_error(test_trues_orig[:, i], test_preds_orig[:, i]) for i in range(test_trues_orig.shape[1])]
            smape_test = symmetric_mean_absolute_percentage_error(test_trues_orig, test_preds_orig)

            # 결과 저장
            test_results_df = pd.DataFrame({'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test, 'MAPE': mape_test, 'SMAPE': smape_test})

            # 결과 저장(for each file)
            test_file_name = os.path.splitext(os.path.basename(test_file))[0]
            results_dir = os.path.join(f"../Results", model_type, group_name, test_file_name)
            os.makedirs(results_dir, exist_ok=True)
            test_results_df.to_csv(os.path.join(results_dir, f'test_result_{test_file_name}_{run_idx}.csv'), index=False)

            # 결과 출력
            print(f"\nTestFile={test_file_name}...")

            target_names = ['CO', 'CO2', 'CH4', 'H2']
            # 결과 저장(summary: all of files)
            with open(summary_results_path, 'a') as f:
                for i, target in enumerate(target_names):
                    mse = mse_test[i]
                    mae = mae_test[i]
                    r2 = r2_test[i]
                    mape = mape_test[i]
                    smape = smape_test[i]
                    f.write(f"{test_file_name},{target},{mse:.4f},{mae:.4f},{r2:.4f},{mape:.4f},{smape:.4f}\n")

            print("[Test] Target-wise metrics:")
            for i, tname in enumerate(target_names):
                y_true_i = test_trues_orig[:, i]
                y_pred_i = test_preds_orig[:, i]

                mse_i = np.mean((y_true_i - y_pred_i) ** 2)
                mae_i = mean_absolute_error(y_true_i, y_pred_i)
                r2_i = r2_score(y_true_i, y_pred_i)
                mape_i = mean_absolute_percentage_error(y_true_i, y_pred_i)
                smape_i = symmetric_mean_absolute_percentage_error(y_true_i, y_pred_i)

                print(f"--- Target: {tname} ---")
                print(f"  MSE : {mse_i:.4f}")
                print(f"  MAE : {mae_i:.4f}")
                print(f"  R2  : {r2_i:.4f}")
                print(f"  MAPE: {mape_i:.2f}%")
                print(f"  SMAPE: {smape_i[0]:.2f}%")
                print("---------------------")

            # 시각화: 테스트 데이터
            time_steps_test = range(len(test_trues_orig))
            plt.figure(figsize=(20, 15))
            for i, tname in enumerate(target_names):
                plt.subplot(2, 2, i + 1)
                plt.plot(time_steps_test, test_trues_orig[:, i], label='Actual')
                plt.plot(time_steps_test, test_preds_orig[:, i], label='Predicted')
                plt.title(f"{tname} - Test Predicted vs Actual")
                plt.xlabel("Elapse(min.)")
                plt.ylabel(tname)
                plt.legend()
                plt.grid(True)
            plt.tight_layout()

            # Figure 저장 경로
            base_fig_dir = os.path.join(f"../Fig", model_type, group_name, test_file_name)
            os.makedirs(base_fig_dir, exist_ok=True)
            fig_file_name = f"plot_{test_file_name}_{run_idx}.png"
            plt.savefig(os.path.join(base_fig_dir, fig_file_name))
            plt.close()
        writer.close()

    # 타겟별 SHAP 평균 및 표준편차 계산
    shap_summary_dir = os.path.join(f'../SHAP', model_type, 'Summary')
    os.makedirs(shap_summary_dir, exist_ok=True)

    for target in ["CO", "CO2", "CH4", "H2"]:
        shap_values_array = np.array(shap_values_list[target])  # (10(반복수), 샘플, 타임스텝, 피처)
        mean_shap_values = np.mean(np.abs(shap_values_array), axis=(0, 1, 2))  # (피처 개수: 20) - feature_t-n의 중요도를 보겠다기 보다는 feature 자체의 중요도
        std_shap_values = np.std(np.abs(shap_values_array), axis=(0, 1, 2))  # (피처 개수: 20) - 만약 어느 시점이 중요한지도 알고 싶다면 axis = (0, 1)로 설정

        feature_names = args.feature_names

        # 피처 중요도 데이터프레임 생성 및 저장 (평균 ± 표준편차)
        shap_feature_summary_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Mean": mean_shap_values,
            "SHAP Std": std_shap_values
        }).sort_values(by="SHAP Mean", ascending=False)

        shap_feature_summary_df.to_csv(os.path.join(shap_summary_dir, f"shap_feature_importance_summary_{target}.csv"),
                                       index=False)

        # SHAP 피처 중요도 시각화 (평균 ± 표준편차 바 그래프)
        plt.figure(figsize=(12, 6))
        plt.barh(shap_feature_summary_df["Feature"], shap_feature_summary_df["SHAP Mean"],
                 xerr=shap_feature_summary_df["SHAP Std"], color="skyblue", capsize=5)
        plt.xlabel("Mean |SHAP Value| ± Std")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance Based on SHAP (Train Data) - {target} (10 Runs)")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(shap_summary_dir, f"shap_feature_importance_summary_{target}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for training a model with different hyperparameters.")
    parser.add_argument('--model_type', type=str, default='rnn', help="Type of model to train")
    parser.add_argument('--groups', nargs='+', type=str,
                        default=['air gasification',
                                 'steam gasification',
                                 'CO2 gasification',
                                 'O2steam gasification'], help="Group to train")
    parser.add_argument('--num_epoch', type=int, default=100, help="Number of epochs")
    parser.add_argument('--num_runs', type=int, default=10, help="Number of runs")
    parser.add_argument('--feature_names', nargs='+', type=str, default=[
        'Actual feeding rate [g/min]',
        'Total N2 flow rate [g/min]',
        'O2 flow rate [g/min]',
        'CO2 flow rate [g/min]',
        'Steam flow rate [g/min]',
        'FNC-1', 'FNC-2', 'FNC-3',
        'TI-2', 'TI-3', 'TI-4', 'TI-5',
        'PI-2', 'PI-3', 'PI-4', 'PI-5',
        'CO(t-1)', 'CO2(t-1)', 'CH4(t-1)', 'H2(t-1)'], help="Feature names to use")
    args = parser.parse_args()

    main()