# Gasification 데이터를 하나의 그룹으로 취급하여 학습 및 테스트하는 코드, Group 1은 따로.
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import optuna
import datetime
# 사용자 정의 모델들
from models.vanilla_rnn import VanillaRNN
from models.vanilla_lstm import VanillaLSTM
from models.vanilla_gru import VanillaGRU

# 사용자 정의 메트릭들
from utils.metrics import symmetric_mean_absolute_percentage_error

# 사용자 정의 함수들
from utils.functions import train_model
from utils.functions import evaluate_model


# 기타
from utils.functions import get_csv_files

# 데이터셋
from datasetes.dataset_rnn import RNNTimeSeriesDataset
from preprocessing.data_prepro import abnormal_0, abnormal_1  # 리스트 임포트
from preprocessing.data_prepro import load_csv_with_issue_handling
from preprocessing.data_prepro import get_window_data_for_rnn

# 평가 지표
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def objective(trial):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_start_time = time.time()  # Trial 시작 시간
    batch_size = trial.suggest_categorical('batch_size', args.batch_size)
    lr = trial.suggest_categorical('lr', args.lr)
    window_size = trial.suggest_categorical('window_size', args.window_size)
    hidden_dim = trial.suggest_categorical('hidden_dim', args.hidden_dim)
    num_epochs = args.num_epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_type = args.model_type
    groups = args.groups

    trial.set_user_attr("model", model_type)
    trial.set_user_attr("group", groups)
    trial.set_user_attr("timestamp", timestamp)

    print(f"Model Type: {model_type}, batch_size: {batch_size}, lr: {lr}, window_size: {window_size}, hidden_dim: {hidden_dim}, num_epochs: {num_epochs}, group: {groups}")

    # 공통 디렉토리 이름 생성 (하이퍼파라미터 기준)
    dir_name = f"{model_type}_batch_{batch_size}_lr_{lr}_window_{window_size}_hidden_{hidden_dim}"

    r2_score_list = []

    for run_idx in range(4): # 4번 반복
        # 그룹별 train/test 파일 선정
        train_files, test_files = [], []
        for group in args.groups:
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
            if df_train is None or len(df_train) < window_size:
                continue
            X_, y_ = get_window_data_for_rnn(df_train, window_size, scale_data=False)
            if X_ is not None:
                train_data_X.append(X_)
                train_data_y.append(y_)
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

        # validation 데이터 준비
        val_data_X, val_data_y = [], []
        for _, test_file in test_files:
            df_test = load_csv_with_issue_handling(test_file)
            if df_test is not None and len(df_test) >= window_size:
                X_test, y_test = get_window_data_for_rnn(df_test, window_size, scale_data=False)
                if X_test is not None:
                    val_data_X.append(X_test)
                    val_data_y.append(y_test)

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
        run_dir = os.path.join(f"runs/{args.vanilla_code}", f'{dir_name}_{timestamp}', "integration", f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # TensorBoard writer
        writer = SummaryWriter(run_dir)

        # 모델 초기화 (매 run_idx마다 새로 생성)
        if model_type == 'rnn':
            model = VanillaRNN(20, hidden_dim, 4).to(device)
        elif model_type == 'lstm':
            model = VanillaLSTM(20, hidden_dim, 4).to(device)
        elif model_type == 'gru':
            model = VanillaGRU(20, hidden_dim, 4).to(device)
        else:
            raise ValueError("Unknown model type.")

        # Loss 및 Optimizer 설정
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        patience = 10
        counter = 0
        best_val_loss = float('inf')

        # 학습 시작
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

        # best 모델 불러오기
        best_model_path = os.path.join(checkpoints_dir, f"best.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        scaler_X = checkpoint['scaler_X']
        scaler_y = checkpoint['scaler_y']

        # test 데이터 준비
        for group_name, test_file in test_files:
            df_test = load_csv_with_issue_handling(test_file)
            if df_test is None or len(df_test) < window_size:
                print("[Skip] test is empty.")
                continue
            X_test, y_test = get_window_data_for_rnn(df_test, window_size, scale_data=False)
            if X_test is None:
                print("[Skip] no test data.")
                continue

            # test 데이터 스케일링 및 Dataset, DataLoader 생성
            N_test, T_test, D_test = X_test.shape
            X_test_2d = X_test.reshape(N_test * T_test, D_test)
            X_test_2d_scaled = scaler_X.transform(X_test_2d)
            X_test_scaled = X_test_2d_scaled.reshape(N_test, T_test, D_test)
            y_test_scaled = scaler_y.transform(y_test)

            test_dataset = RNNTimeSeriesDataset(X_test_scaled, y_test_scaled)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            test_loss_scaled, test_trues_scaled, test_preds_scaled = evaluate_model(model, test_loader, criterion, device, epoch, writer=None, phase="Test")

            test_trues_orig = scaler_y.inverse_transform(test_trues_scaled)
            test_preds_orig = scaler_y.inverse_transform(test_preds_scaled)

            # R2 score 계산
            r2_scores = [r2_score(test_trues_orig[:, i], test_preds_orig[:, i]) for i in range(test_trues_orig.shape[1])]
            r2_score_list.append(r2_scores)

            # 결과 저장
            test_results_df = pd.DataFrame({
                'MAE': mean_absolute_error(test_trues_orig, test_preds_orig, multioutput='raw_values').tolist(),
                'MSE': mean_squared_error(test_trues_orig, test_preds_orig, multioutput='raw_values').tolist(),
                'R2': r2_scores,
                'SMAPE': symmetric_mean_absolute_percentage_error(test_trues_orig, test_preds_orig)
            })
            # 결과 저장 경로
            base_test = os.path.basename(test_file)
            test_file_name = os.path.splitext(base_test)[0]
            results_dir = os.path.join(f"../Results/{args.vanilla_code}", dir_name, "integration", group_name, f'{test_file_name}_{timestamp}')
            os.makedirs(results_dir, exist_ok=True)
            test_results_df.to_csv(os.path.join(results_dir, f'test_results{timestamp}.csv'), index=False)

            # 결과 출력
            print(f"\nTestFile={base_test}...")

            target_names = ['CO', 'CO2', 'CH4', 'H2']
            print("[Test: Original Scale] Target-wise metrics:")
            for i, tname in enumerate(target_names):
                y_true_i = test_trues_orig[:, i]
                y_pred_i = test_preds_orig[:, i]

                mse_i = np.mean((y_true_i - y_pred_i) ** 2)
                mae_i = mean_absolute_error(y_true_i, y_pred_i)
                r2_i = r2_score(y_true_i, y_pred_i)
                smape_i = symmetric_mean_absolute_percentage_error(y_true_i, y_pred_i)

                print(f"--- Target: {tname} ---")
                print(f"  MSE : {mse_i:.4f}")
                print(f"  MAE : {mae_i:.4f}")
                print(f"  R2  : {r2_i:.4f}")
                print(f"  SMAPE: {smape_i[0]:.2f}%")
                print("---------------------")

            # 시각화: 테스트 데이터
            time_steps_test = range(len(test_trues_orig))
            plt.figure(figsize=(20, 15))
            for i, tname in enumerate(target_names):
                plt.subplot(2, 2, i+1)
                plt.plot(time_steps_test, test_trues_orig[:, i], label='Actual')
                plt.plot(time_steps_test, test_preds_orig[:, i], label='Predicted')
                plt.title(f"{tname} - Test Prediction vs Actual")
                plt.xlabel("Elapse(min.)")
                plt.ylabel(tname)
                plt.legend()
                plt.grid(True)
            plt.tight_layout()

            # Figure 저장 경로
            base_fig_dir = os.path.join(f"../Fig/{args.vanilla_code}", dir_name, "integration", group_name,f'{test_file_name}_{timestamp}')
            os.makedirs(base_fig_dir, exist_ok=True)
            fig_file_name = f"plot_{timestamp}.png"
            plt.savefig(os.path.join(base_fig_dir, fig_file_name))
            plt.close()

        writer.close()
        # end of test_file loop
    # end of run_idx loop
    # Objective 값 반환: Mean |1 - R2| + λ * Std(R2))
    lambda_ = 0.1
    r2_score_array = np.array(r2_score_list)  # 리스트를 배열로 변환
    mean_r2_error = np.mean(np.abs(1 - r2_score_array))
    std_r2 = np.std(r2_score_array)
    print(f"Mean |1 - R2|: {mean_r2_error:.4f}, Std(R2): {std_r2:.4f}")

    # Trial 소요 시간 측정 및 예상 남은 시간 출력
    trial_duration = time.time() - trial_start_time
    elapsed_trials = len(trial.study.trials)
    remaining_trials = args.n_trials - elapsed_trials

    # 예상 남은 시간 계산
    estimated_total_time = trial_duration * args.n_trials
    estimated_remaining_time = trial_duration * remaining_trials

    print(f"Trial {elapsed_trials}/{args.n_trials} completed in {trial_duration:.2f} seconds.")
    print(f"Estimated Total Time: {str(datetime.timedelta(seconds=estimated_total_time))}")
    print(f"Estimated Remaining Time: {str(datetime.timedelta(seconds=estimated_remaining_time))}")

    return mean_r2_error + lambda_ * std_r2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A script for training a model with different hyperparameters.")
    # 변경해야할 하이퍼파라미터: model_type
    parser.add_argument('--model_type', type=str, default='gru', help="Model type to train: rnn, lstm, gru")
    parser.add_argument('--groups', nargs='+', type=str,
                        default=['Group 2_air gasification',
                                 'Group 3_steam gasification',
                                 'Group 4_CO2 gasification',
                                 'Group 5_O2steam gasification'], help="Group to train")
    parser.add_argument('--batch_size', nargs='+', type=int, default=[16, 32, 64, 128], help="Batch size for training")
    parser.add_argument('--lr', nargs='+', type=float, default=[1e-4, 1e-3, 1e-2], help="Learning rate")
    parser.add_argument('--window_size', nargs='+', type=int, default=[5, 10, 20, 30], help="Window size for sequence data")
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[128, 256], help="Hidden dimension size")
    parser.add_argument('--num_epoch', type=int, default=100, help="Number of epochs")
    parser.add_argument('--n_trials', type=int, default=60, help="Number of Optuna trials")
    parser.add_argument('--study_name', type=str, default="study", help="Optuna study name")
    parser.add_argument('--storage', type=str, default=None, help="Database URL for Optuna study (optional)")
    parser.add_argument("--vanilla_code", type=str, default="vanilla3", help="vanilla code number")
    args = parser.parse_args()

    # 전체 실행 시간 측정
    script_start_time = time.time()

    # SQLite DB 파일 경로
    storage = optuna.storages.RDBStorage(
        url=f'sqlite:///{os.path.join(f"../databases/{args.vanilla_code}", f"{args.model_type}_integration.db")}',
        engine_kwargs={
            'pool_size': 5,
            'connect_args': {
                'timeout': 15
            }
        }
    )

    try:
        study = optuna.load_study(study_name='my_study', storage=storage)
    except KeyError:
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True  # DB 사용 시 기존 study를 불러옴
        )

    print("Starting Optuna optimization...")
    study_start_time = time.time()  # Optuna 시작 시간
    study.optimize(objective, n_trials=args.n_trials)
    study_end_time = time.time()  # Optuna 종료 시간
    print(f"Optuna optimization completed in {str(datetime.timedelta(seconds=study_end_time - study_start_time))}")

    # 전체 실행 시간 출력
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    print(f"\nTotal script execution time: {str(datetime.timedelta(seconds=total_script_time))}")

    # 최적 하이퍼파라미터 출력
    print("\nBest trial:")
    print(f"Value: {study.best_trial.value}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # 최적화 결과 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"optuna_results/{args.vanilla_code}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"optuna_results_{args.model_type}_integration_{timestamp}.csv")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(results_file, index=False)
    print(f"\nOptuna results saved to {results_file}")