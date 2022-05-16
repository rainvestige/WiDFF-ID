# -* coding:utf-8 -*
"""
# the author: {zs}
# the date: {} 
"""
import torch
import torch.nn as nn

####################定义RNN类##############################################
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(lstm, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, 42)

    def forward(self, x):
        r_out,_= self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        return out, out
