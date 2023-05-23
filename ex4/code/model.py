import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = 256
        self.output_size = output_size
        self.input_size = input_size
        self.input2hidden = nn.Linear(self.input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.hidden2output = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def __str__(self):
        return 'rnn'

    def forward(self, x, pre_state=None):
        batch_size, seq_length, _ = x.size()
        if pre_state is None:
            pre_state = torch.zeros(batch_size, self.hidden_size).to(self.device)
        output = torch.zeros(batch_size, self.output_size).to(self.device)
        for i in range(seq_length):
            token = x[:, i, :].to(self.device)
            combined = torch.cat((token, pre_state), 1).to(self.device)
            pre_state = self.input2hidden(combined).to(self.device)
            pre_state = self.tanh(pre_state)    .to(self.device)
            output = self.hidden2output(pre_state).to(self.device)
        return output


class LSTM(nn.Module):

    def __init__(self, input_size, output_size,  bidirectional=False):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = 256
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.forget_gate = nn.Linear(input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.input_gate = nn.Linear(input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.c_gate = nn.Linear(input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.output_gate = nn.Linear(input_size + self.hidden_size, self.hidden_size).to(self.device)
        if not bidirectional:
            self.hidden2output = nn.Linear(self.hidden_size, output_size).to(self.device)
        else:
            self.hidden2output = nn.Linear(self.hidden_size * 2, output_size).to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    def __str__(self):
        if self.bidirectional:
            return 'bi-lstm'
        else:
            return 'lstm'

    def forward(self, x):
        x = x.to(torch.double)
        batch_size, seq_length, _ = x.size()
        if not self.bidirectional:
            hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
            ct = torch.zeros(batch_size, self.hidden_size).to(self.device)
            result = torch.zeros(batch_size, self.output_size).to(self.device)
            for i in range(seq_length):
                token = x[:, i, :].to(self.device).to(torch.double)
                combined = torch.cat((token, hidden), 1).to(torch.double)
                forget = self.sigmoid(self.forget_gate(combined)).to(torch.double)
                input_ = self.sigmoid(self.input_gate(combined)).to(torch.double)
                c_ = self.tanh(self.c_gate(combined)).to(torch.double)
                output = self.sigmoid(self.output_gate(combined)).to(torch.double)
                ct = ct * forget + input_ * c_
                hidden = self.tanh(ct) * output
                result = self.hidden2output(hidden)
            return result
        else:
            hidden1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
            hidden2 = torch.zeros(batch_size, self.hidden_size).to(self.device)
            hidden1s = []
            hidden2s = []
            ct1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
            ct2 = torch.zeros(batch_size, self.hidden_size).to(self.device)
            for i in range(seq_length):
                token1 = x[:, i, :].to(self.device)
                token2 = x[:, seq_length-i-1, :].to(self.device)
                combined1 = torch.cat((token1, hidden1), 1)
                combined2 = torch.cat((token2, hidden2), 1)
                forget1 = self.sigmoid(self.forget_gate(combined1))
                forget2 = self.sigmoid(self.forget_gate(combined2))
                input1 = self.sigmoid(self.input_gate(combined1))
                input2 = self.sigmoid(self.input_gate(combined2))
                c_1 = self.tanh(self.c_gate(combined1))
                c_2 = self.tanh(self.c_gate(combined2))
                output1 = self.sigmoid(self.output_gate(combined1))
                output2 = self.sigmoid(self.output_gate(combined2))
                ct1 = ct1 * forget1 + input1 * c_1
                ct2 = ct2 * forget2 + input2 * c_2
                hidden1 = self.tanh(ct1) * output1
                hidden2 = self.tanh(ct2) * output2
                hidden1s.append(hidden1)
                hidden2s.insert(0, hidden2)
            hidden1 = torch.stack(hidden1s).mean(0)
            hidden2 = torch.stack(hidden2s).mean(0)
            result = self.hidden2output(torch.cat((hidden1, hidden2), 1))
            return result


class GRU(nn.Module):

    def __str__(self):
        return 'gru'

    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = 256
        self.output_size = output_size
        self.input_size = input_size
        self.reset_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.update_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.h_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size).to(self.device)
        self.hidden2output = nn.Linear(self.hidden_size, output_size).to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        ones = torch.ones(batch_size, self.hidden_size).to(self.device)
        output = torch.zeros(batch_size, self.output_size).to(self.device)
        for i in range(seq_length):
            token = x[:, i, :]
            combined = torch.cat((token, hidden), 1).to(self.device)
            reset = self.sigmoid(self.reset_gate(combined)).to(self.device)
            zt = self.sigmoid(self.update_gate(combined)).to(self.device)
            combined2 = torch.cat((token, reset * hidden), 1).to(self.device)
            h_ = self.tanh(self.h_gate(combined2)).to(self.device)
            hidden = zt * hidden + (ones - zt) * h_
            output = self.hidden2output(hidden)
        return output
