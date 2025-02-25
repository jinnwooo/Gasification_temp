# 특정 그룹의 파일을 평가하는 스크립트
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
import json
import pandas as pd

# 모델들
from models.vanilla_rnn import VanillaRNN
from models.vanilla_lstm import VanillaLSTM
from models.vanilla_gru import VanillaGRU

# 사용자 정의 메트릭들
from utils.metrics import symmetric_mean_absolute_percentage_error

# 사용자 정의 함수들
from utils.functions import evaluate_model, calculate_shap_values, determine_sample_size
from preprocessing.data_prepro import load_csv_with_issue_handling, get_window_data_for_rnn
from datasetes.dataset_rnn import RNNTimeSeriesDataset

# 평가 지표
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 타입
    model_type = args.model_type
    if model_type == 'rnn':
        parameters = {"model_type": 'rnn', "batch_size": 32, "lr": 0.001, "window_size": 10, "hidden_dim": 256}
        model = VanillaRNN(20, parameters['hidden_dim'], 4).to(device)

    elif model_type == 'lstm':
        parameters = {"model_type": 'lstm', "batch_size": 16, "lr": 0.001, "window_size": 10, "hidden_dim": 256}
        model = VanillaLSTM(20, parameters['hidden_dim'], 4).to(device)

    elif model_type == 'gru':
        parameters = {"model_type": 'gru', "batch_size": 32, "lr": 0.001, "window_size": 20, "hidden_dim": 128}
        model = VanillaGRU(20, parameters['hidden_dim'], 4).to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Choose from ['rnn', 'lstm', 'gru']")

    summary_results_dir = os.path.join(f"../Results/{model_type}")
    os.makedirs(summary_results_dir, exist_ok=True)
    summary_results_path = os.path.join(summary_results_dir, "summary_results.csv")

    if not os.path.exists(summary_results_path) or os.stat(summary_results_path).st_size == 0:
        df_header = pd.DataFrame(columns=["Test_File", "Target", "MSE", "MAE", "R2", "SMAPE"])
        df_header.to_csv(summary_results_path, index=False, encoding="utf-8-sig")

    shap_values_list = {target: [] for target in ["CO", "CO2", "CH4", "H2"]}

    for run_idx in range(10):
        # run_folder 로드: 모델 파라미터, 체크포인트, 스케일러, 테스트 파일
        # run_dir 설정
        run_dir = os.path.join("runs", model_type, f"run_{run_idx}")

        # test_files.json 경로
        test_file_path = os.path.join(run_dir, "test_files.json")

        # test_files.json 로드
        if os.path.exists(test_file_path):
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_files = json.load(f)  # 리스트 형태로 저장됨

            print(f"Run {run_idx}: 테스트 파일 로드 완료")
            for test_entry in test_files:
                print(f"  그룹: {test_entry['group']}, 테스트 파일: {test_entry['test_file']}")
        else:
            print(f"Run {run_idx}: test_files.json 파일이 존재하지 않습니다.")

        checkpoints_dir = os.path.join(run_dir, "checkpoints")

        best_model_path = os.path.join(checkpoints_dir, "best.pth")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']) # 가중치 불러오기
        epoch = checkpoint['epoch']
        scaler_X = checkpoint['scaler_X']
        scaler_y = checkpoint['scaler_y']

        criterion = nn.L1Loss()

        # test data evaluate
        for num, test_entry  in enumerate(test_files):
            group_name, test_file = test_entry["group"], test_entry["test_file"]
            df_test = load_csv_with_issue_handling(test_file)
            X_test, y_test = get_window_data_for_rnn(df_test, parameters['window_size'], scale_data=False)

            N_test, T_test, D_test = X_test.shape
            X_test_2d = X_test.reshape(N_test * T_test, D_test)
            X_test_2d_scaled = scaler_X.transform(X_test_2d)
            X_test_scaled = X_test_2d_scaled.reshape(N_test, T_test, D_test)
            y_test_scaled = scaler_y.transform(y_test)

            test_dataset = RNNTimeSeriesDataset(X_test_scaled, y_test_scaled)
            test_loader = DataLoader(test_dataset, batch_size=parameters['batch_size'], shuffle=False)

            test_loss, test_trues, test_preds = evaluate_model(model, test_loader, criterion, device, epoch,
                                                               writer=None, phase="Test")

            test_trues_orig = scaler_y.inverse_transform(test_trues)
            test_preds_orig = scaler_y.inverse_transform(test_preds)

            # 성능 지표 계산
            mae_test = mean_absolute_error(test_trues_orig, test_preds_orig, multioutput='raw_values').tolist()
            mse_test = mean_squared_error(test_trues_orig, test_preds_orig, multioutput='raw_values').tolist()
            r2_test = [r2_score(test_trues_orig[:, i], test_preds_orig[:, i]) for i in range(test_trues_orig.shape[1])]
            smape_test = symmetric_mean_absolute_percentage_error(test_trues_orig, test_preds_orig)

            # 결과 저장
            test_results_df = pd.DataFrame({'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test, 'SMAPE': smape_test})

            # 결과 저장(for each file)
            test_file_name = os.path.splitext(os.path.basename(test_file))[0]
            results_dir = os.path.join(f"../Results", model_type, group_name, test_file_name)
            os.makedirs(results_dir, exist_ok=True)
            test_results_df.to_csv(os.path.join(results_dir, f'test_result_{test_file_name}_{run_idx}.csv'),
                                   index=False, encoding='utf-8-sig')

            # 결과 출력
            print(f"\nTestFile={test_file_name}...")

            target_names = ['CO', 'CO2', 'CH4', 'H2']
            # 결과 저장(summary: all of files)
            with open(summary_results_path, 'a', encoding='utf-8-sig') as f:
                for i, target in enumerate(target_names):
                    mse = mse_test[i]
                    mae = mae_test[i]
                    r2 = r2_test[i]
                    smape = smape_test[i]
                    f.write(f"{test_file_name},{target},{mse:.4f},{mae:.4f},{r2:.4f},{smape:.4f}\n")

            print("[Test] Target-wise metrics:")
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
            target_names = [r"$\mathrm{CO}$", r"$\mathrm{CO_2}$", r"$\mathrm{CH_4}$", r"$\mathrm{H_2}$"]
            plt.figure(figsize=(20, 15), dpi=600)
            for i, tname in enumerate(target_names):
                plt.subplot(2, 2, i + 1)
                # 실제값
                plt.plot(time_steps_test, test_trues_orig[:, i], label='Actual', color='black', linestyle='-',
                         linewidth=3, alpha=0.8)

                # 예측값 (같은 실선 but 연하게)
                plt.plot(time_steps_test, test_preds_orig[:, i], label='Predicted', color='orange', linestyle = '--',
                         alpha=0.6)

                # x축, y축 설정
                plt.xlabel("Time (s)", fontsize=20)
                plt.ylabel(tname, fontsize=20)

                # 그리드 스타일
                plt.grid(True, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

                # 범례
                plt.legend(loc="lower right", fontsize=20, frameon=False)

                plt.xticks(fontsize=18)  # x축 숫자 크기 키우기
                plt.yticks(fontsize=18)  # y축 숫자 크기 키우기

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Figure 저장 경로
            base_fig_dir = os.path.join(f"../Fig", model_type, group_name, test_file_name)
            os.makedirs(base_fig_dir, exist_ok=True)
            fig_file_name = f"plot_{test_file_name}_{run_idx}.png"
            plt.savefig(os.path.join(base_fig_dir, fig_file_name))
            plt.close()

            # SHAP
            print(f"SHAP 계산 중... {num+1} / 4 | {run_idx} / 9")
            sample_size = determine_sample_size(test_loader)
            print(f"SHAP 샘플링 개수: {sample_size}")
            shap_values, X_sample = calculate_shap_values(model, test_loader, device)
            shap_values_original_scale = shap_values * scaler_X.scale_.reshape(1, 1, 20, 1)

            # SHAP 결과 저장 디렉토리 생성
            shap_results_dir = os.path.join(f"../SHAP", model_type, "Results", f"run_{run_idx}", test_file_name)
            os.makedirs(shap_results_dir, exist_ok=True)
            # 타겟별 SHAP 값 저장 (Numpy 파일)
            for i, target in enumerate(["CO", "CO2", "CH4", "H2"]):
                shap_values_target = shap_values_original_scale[:, :, :, i]  # 특정 타겟에 대한 SHAP 값 추출
                np.save(os.path.join(shap_results_dir, f"shap_values_train_{target}_{run_idx}.npy"), shap_values_target)
                shap_values_list[target].append(shap_values_target)
            print(f"Test file {num+1} / 4 완료")
        print(f"Run {run_idx} 완료")

    # 타겟별 SHAP 평균 및 표준편차 계산
    shap_summary_dir = os.path.join(f'../SHAP', model_type, 'Summary')
    os.makedirs(shap_summary_dir, exist_ok=True)

    for target in ["CO", "CO2", "CH4", "H2"]:
        shap_values_array = np.array(shap_values_list[target])  # (40(반복수*테스트 파일수), 샘플, 타임스텝, 피처)
        mean_shap_values = np.mean(np.abs(shap_values_array),
                                   axis=(0, 1, 2))  # (피처 개수: 20) - feature_t-n의 중요도를 보겠다기 보다는 feature 자체의 중요도
        std_shap_values = np.std(np.abs(shap_values_array),
                                 axis=(0, 1, 2))  # (피처 개수: 20) - 만약 어느 시점이 중요한지도 알고 싶다면 axis = (0, 1)로 설정

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
        plt.figure(figsize=(12, 6), dpi=300)
        plt.barh(shap_feature_summary_df["Feature"], shap_feature_summary_df["SHAP Mean"],
                 xerr=shap_feature_summary_df["SHAP Std"], color="skyblue", capsize=5)
        plt.xlabel("Mean |SHAP Value| ± Std")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance Based on SHAP - {target}")
        plt.gca().invert_yaxis()
        plt.yticks(fontsize=8, rotation=0)
        plt.savefig(os.path.join(shap_summary_dir, f"shap_feature_importance_summary_{target}.png"))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script for evaluating, visualizing a model and calculating SHAP.")
    parser.add_argument("--model_type", type=str, default="gru", help="Model type to evaluate")
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
