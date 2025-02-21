import torch
from torch.utils.data import Dataset

class RNNTimeSeriesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        (X, y) 배열을 직접 넣어주면 Dataset이 그것을 그대로 관리하는 방식
        window_size: 슬라이딩 윈도우 크기
        transform: 필요하면
        """
        super().__init__()
        self.X = X
        self.y = y
        self.transform = transform

        # 간단히 체크
        if len(self.X) != len(self.y):
            raise ValueError("X와 y의 샘플 수가 다릅니다.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx]            # (seq_len, feature_dim)
        y_sample = self.y[idx]            # (target_dim(4),)

        if self.transform:
            X_sample, y_sample = self.transform(X_sample, y_sample)

        # numpy -> torch
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_sample = torch.tensor(y_sample, dtype=torch.float32)

        return X_sample, y_sample
