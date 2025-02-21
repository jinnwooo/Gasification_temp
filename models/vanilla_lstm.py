import torch
import torch.nn as nn
import torch.nn.init as init

class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.final_fc = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # LSTM 초기화
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)  # Xavier 초기화
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal 초기화
            elif 'bias' in name:
                init.zeros_(param.data)  # bias 0으로 초기화

        # FC 레이어 초기화
        init.xavier_uniform_(self.final_fc.weight)
        init.zeros_(self.final_fc.bias)

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, input_dim)
        lstm_out, (h, c) = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]  # 마지막 타임스텝

        out = self.final_fc(lstm_feat)
        return out
