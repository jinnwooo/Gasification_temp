import torch
import torch.nn as nn
import torch.nn.init as init

class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        self.rnn1 = nn.RNN(input_dim, hidden_dim, batch_first=True)
        # self.rnn2 = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.final_fc = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # RNN 초기화
        for name, param in self.rnn1.named_parameters():
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
        rnn_out1, hidden1 = self.rnn1(x_seq)
        # rnn_out1: (batch, seq_len, hidden_dim)
        # hidden1: (1, batch, hidden_dim) for 1-layer RNN

        # 마지막 타임스텝의 출력만 사용
        rnn_feat = rnn_out1[:, -1, :]     # (batch, hidden_dim)

        out = self.final_fc(rnn_feat)
        return out
