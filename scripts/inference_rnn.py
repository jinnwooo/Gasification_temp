# 특정 그룹의 파일을 평가하는 스크립트
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader

# 모델들
from models.vanilla_rnn import VanillaRNN
from models.vanilla_lstm import VanillaLSTM
from models.vanilla_gru import VanillaGRU

# 사용자 정의 메트릭들
from utils.metrics import mean_absolute_percentage_error, smape

# 사용자 정의 함수들
from utils.functions import evaluate_model

# (수정) CSV -> (X, y) 전처리 함수 (스케일 전 상태로)
from preprocessing.data_prepro import load_csv_with_issue_handling, get_window_data_for_rnn

# (수정) Dataset: (X, y) numpy array로부터 구성
from datasetes.dataset_rnn import RNNTimeSeriesDataset

# 평가 지표
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # 예시 run_folder
    run_folder = r"C:\Lab\TimeSeries\scripts\runs\vanilla3\gru_batch_32_lr_0.001_window_10_hidden_256_20250215_023316\integration\run_0"
    checkpoints_dir = os.path.join(run_folder, "checkpoints")

    # 모델 타입
    model_type = "gru"  # 'rnn', 'lstm', 'gru'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Run Folder: 20210614_non-A.C_air가스화.csv")

    # (1) 체크포인트 로드 (scaler_X, scaler_y 포함)
    ckpt_filename = "best.pth"
    best_ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    if not os.path.exists(best_ckpt_path):
        print(f"{ckpt_filename} not found in {checkpoints_dir}")
        sys.exit(1)

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    scaler_X = checkpoint.get('scaler_X', None)  # 입력 스케일러
    scaler_y = checkpoint.get('scaler_y', None)  # 타겟 스케일러

    epoch_loaded = checkpoint.get('epoch', -1)
    best_val_loss = checkpoint.get('val_loss', None)
    print(f"Loaded checkpoint from epoch={epoch_loaded}, best_val_loss={best_val_loss}")

    # (2) 모델 하이퍼파라미터 (학습 시와 동일)
    input_dim = 20
    hidden_dim = 256
    output_dim = 4
    window_size = 10

    # (3) 모델 초기화
    if model_type == 'vanilla':
        model = VanillaRNN(input_dim, hidden_dim, output_dim)
    elif model_type == 'lstm':
        model = VanillaLSTM(input_dim, hidden_dim, output_dim)
    elif model_type == 'gru':
        model = VanillaGRU(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError("Unknown model type.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # (4) 테스트 CSV 파일 (Path(run_folder).name.csv)
    test_csv_name = "20210614_non-A.C_air가스화.csv"
    test_csv_path = os.path.join("../data/Gasification", test_csv_name)
    if not os.path.exists(test_csv_path):
        print("No test.csv found. Please specify or parse from run_folder.")
        sys.exit(1)

    # (5) (X_test, y_test) 로드 (원본 스케일)
    df_test = load_csv_with_issue_handling(test_csv_path)
    if df_test is None:
        print("No valid test data. Check your CSV.")
        sys.exit(1)

    X_test, y_test = get_window_data_for_rnn(df_test, window_size=window_size, scale_data=False)
    if X_test is None or len(X_test) == 0:
        print("test data is empty after windowing.")
        sys.exit(1)

    # (5-1) X_test, y_test를 스케일러로 변환
    #       => 반드시 학습 때 fit된 scaler_X, scaler_y 써야 함
    N_test, T_test, D_test = X_test.shape
    X_test_2d = X_test.reshape(N_test * T_test, D_test)

    if scaler_X is None or scaler_y is None:
        print("No scaler found in checkpoint. The model might have been trained without scaling.")
        print("Proceeding without transform. (Results might be off if training used scaling.)")
        X_test_scaled = X_test
        y_test_scaled = y_test
    else:
        # transform X
        X_test_2d_scaled = scaler_X.transform(X_test_2d)  # (N*T, D)
        X_test_scaled = X_test_2d_scaled.reshape(N_test, T_test, D_test)  # 원래 shape 복원

        # transform y
        y_test_scaled = scaler_y.transform(y_test)  # (N_test,4)

    # (5-2) Dataset, DataLoader 생성
    test_dataset = RNNTimeSeriesDataset(X_test_scaled, y_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.L1Loss()

    # (6) 평가 (스케일된 y 기준 loss)
    test_loss_scaled, all_trues_scaled, all_preds_scaled = evaluate_model(
        model, test_loader, criterion, device, epoch_loaded, writer=None, phase="Test"
    )

    # (7) y를 원본 스케일로 inverse_transform
    all_trues_orig = all_trues_scaled
    all_preds_orig = all_preds_scaled
    if scaler_y:
        all_trues_orig = scaler_y.inverse_transform(all_trues_scaled)
        all_preds_orig = scaler_y.inverse_transform(all_preds_scaled)

    # (7-1) 지표 계산 (원본 스케일)
    target_names = ['CO', 'CO2', 'CH4', 'H2']

    print(f"Test Loss(MSE) [Scaled domain]: {test_loss_scaled:.4f}")
    print("--- Metrics on Original Scale ---")
    for i, tname in enumerate(target_names):
        y_true_i = all_trues_orig[:, i]
        y_pred_i = all_preds_orig[:, i]

        mse_i = np.mean((y_true_i - y_pred_i) ** 2)
        mae_i = mean_absolute_error(y_true_i, y_pred_i)
        r2_i = r2_score(y_true_i, y_pred_i)
        mape_i = mean_absolute_percentage_error(y_true_i, y_pred_i)
        smape_i = smape(y_true_i, y_pred_i)

        print(f"[{tname}] MSE={mse_i:.4f}, MAE={mae_i:.4f}, R2={r2_i:.4f}, MAPE={mape_i:.2f}%, SMAPE={smape_i[0]:.2f}%")

    # (8) 시각화
    time_steps = range(len(all_trues_orig))
    for i, tname in enumerate(target_names):
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, all_trues_orig[:, i], label='Actual', marker='o', linestyle='-')
        plt.plot(time_steps, all_preds_orig[:, i], label='Predicted', marker='x', linestyle='--')
        plt.title(f"{tname} Prediction vs Actual (Original Scale)")
        plt.xlabel("Sample Index")
        plt.ylabel(tname)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_filename = f"{tname}_prediction.png"
        plot_path = os.path.join(run_folder, plot_filename)
        # plt.show(plot_path)
        plt.show()
        plt.close()
        # print(f"Saved plot for {tname} at {plot_path}")


if __name__ == '__main__':
    main()
