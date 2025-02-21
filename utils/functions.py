import torch
import numpy as np
from tqdm import tqdm
import glob
import os
import shap

def train_model(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0

    # tqdm으로 감싸기: desc에 epoch 정보 표시
    train_loader_tqdm = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}", unit="batch", dynamic_ncols=True, leave=True)

    for batch_idx, (X_seq, y) in enumerate(train_loader_tqdm):
        X_seq, y = X_seq.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X_seq)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # tqdm에 현재 Loss를 표시
        train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)

    # TensorBoard 기록 (train loss)
    if writer:
        writer.add_scalar('Loss/Train', epoch_loss, epoch)

    return epoch_loss


def evaluate_model(model, loader, criterion, device, epoch, writer=None, phase="Val"):
    model.eval()
    running_loss = 0.0

    # 예측/실제값 저장
    all_preds, all_trues = [], []

    # tqdm: 진행 상황 표시
    loader_tqdm = tqdm(loader, desc=f"[{phase}] Epoch {epoch}", unit="batch", dynamic_ncols=True, leave=True)

    with torch.no_grad():
        for batch_idx, (X_seq, y) in enumerate(loader_tqdm):
            X_seq, y = X_seq.to(device), y.to(device)
            outputs = model(X_seq)
            loss = criterion(outputs, y)
            running_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())  # (batch, 4)
            all_trues.append(y.cpu().numpy())  # (batch, 4)

            # tqdm에 현재 Loss를 표시
            loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

    loss = running_loss / len(loader)

    # TensorBoard 기록
    if writer:
        writer.add_scalar(f'Loss/{phase}', loss, epoch)

    # 배열 연결
    all_preds = np.concatenate(all_preds, axis=0)  # shape: (N, 4)
    all_trues = np.concatenate(all_trues, axis=0)  # shape: (N, 4)

    return loss, all_trues, all_preds

def get_csv_files(group_path):
    return glob.glob(os.path.join(group_path, '*.csv'))


def determine_sample_size(train_loader, max_sample_size=2000):
    """
    Train 데이터 크기를 기반으로 SHAP 샘플링 개수를 자동 결정하는 함수
    :param train_loader: PyTorch DataLoader (Train 데이터)
    :param max_sample_size: 최대 샘플링 개수 (기본값: 2000)
    :return: 사용할 샘플링 개수
    """
    train_data_size = len(train_loader.dataset)

    # 데이터 크기에 따라 샘플링 수 결정
    if train_data_size <= 5000:
        return train_data_size  # 전체 데이터 사용
    elif train_data_size <= 50000:
        return min(2000, train_data_size)  # 최대 2000개 샘플
    else:
        return min(max_sample_size, train_data_size)  # 최대 max_sample_size 개 샘플


def calculate_shap_values(model, loader, device):
    """
    RNN/LSTM/GRU 모델에서 SHAP 값을 계산하는 함수 (requires_grad 문제 해결)
    :param model: 학습된 모델
    :param loader: Train 데이터셋의 DataLoader
    :param device: CPU or CUDA
    :return: SHAP values, 샘플 데이터
    """
    model.train()  # 🔹 SHAP 계산을 위해 train 모드 유지
    torch.backends.cudnn.enabled = False  # 🔹 CuDNN 비활성화 (backward 문제 방지)

    sample_size = determine_sample_size(loader)  # 자동 샘플링 개수 결정

    # 전체 train_loader에서 데이터 가져오기
    all_data = []

    for X_batch, _ in tqdm(loader, desc="Loading Train Data for SHAP", leave=True):
        all_data.append(X_batch)

    X_all = torch.cat(all_data, dim=0)  # 모든 배치를 하나의 텐서로 합치기

    # 샘플 크기 제한 (랜덤 샘플 선택)
    if X_all.shape[0] > sample_size:
        idx = np.random.choice(X_all.shape[0], sample_size, replace=False)
        X_sample = X_all[idx]
    else:
        X_sample = X_all

    X_sample = X_sample.to(device)

    # 🔹 SHAP 계산을 위해 requires_grad 활성화
    X_sample.requires_grad = True

    # SHAP Explainer 생성 (GradientExplainer 사용)
    explainer = shap.GradientExplainer(model, X_sample)

    # tqdm 적용하여 SHAP 값 계산 진행률 표시
    with tqdm(total=len(X_sample), desc="Computing SHAP Values", leave=True) as pbar:
        for i, x in enumerate(X_sample):
            shap_values = explainer.shap_values(x.unsqueeze(0))  # 개별 샘플 단위로 SHAP 계산
            pbar.update(1)  # tqdm 업데이트

    # 🔹 SHAP 계산 후 requires_grad 비활성화 (성능 최적화)
    X_sample.requires_grad = False

    model.eval()  # 🔹 SHAP 계산이 끝난 후 다시 eval() 모드로 변경
    torch.backends.cudnn.enabled = True  # 🔹 CuDNN 다시 활성화

    return shap_values, X_sample.cpu().numpy()
