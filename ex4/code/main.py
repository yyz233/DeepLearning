from data_process import data_process_shopping
from data_process import data_process_climate
from torch.optim.lr_scheduler import MultiStepLR
from model import GRU
from tensorboardX import SummaryWriter
import torch.nn as nn
from model import RNN
from model import LSTM
import torch
from train import train
from train import train_climate
from train import test

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    now_task = 'climate_lstm'
    if now_task == 'online_shopping_rnn':
        print(now_task)
        epoch = 50
        batch_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader, test_dataloader, va_dataloader, label2name_dict = data_process_shopping(batch_size)
        a = 0
        rnn = RNN(128, 10)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, [20, 30], 0.1)
        writer = SummaryWriter('./result')
        train(rnn, epoch, train_dataloader, criterion, optimizer, writer, 'online_shopping', va_dataloader)
        test(rnn, test_dataloader)
        print('success!')
    if now_task == 'online_shopping_lstm':
        print(now_task)
        epoch = 50
        batch_size = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader, test_dataloader, va_dataloader, label2name_dict = data_process_shopping(batch_size)
        a = 0
        lstm = LSTM(128, 10, False)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, [20, 30], 0.1)
        writer = SummaryWriter('./result')
        train(lstm, epoch, train_dataloader, criterion, optimizer, writer, 'online_shopping', va_dataloader)
        test(lstm, test_dataloader)
        print('success!')
    if now_task == 'online_shopping_bi_lstm':
        print(now_task)
        epoch = 50
        batch_size = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader, test_dataloader, va_dataloader, label2name_dict = data_process_shopping(batch_size)
        a = 0
        bi_lstm = LSTM(128, 10, True)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, [20, 30], 0.1)
        writer = SummaryWriter('./result')
        train(bi_lstm, epoch, train_dataloader, criterion, optimizer, writer, 'online_shopping', va_dataloader)
        test(bi_lstm, test_dataloader)
        print('success!')
    if now_task == 'online_shopping_gru':
        print(now_task)
        epoch = 50
        batch_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader, test_dataloader, va_dataloader, label2name_dict = data_process_shopping(batch_size)
        a = 0
        gru = GRU(128, 10)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(gru.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, [20, 30], 0.1)
        writer = SummaryWriter('./result')
        train(gru, epoch, train_dataloader, criterion, optimizer, writer, 'online_shopping', va_dataloader)
        test(gru, test_dataloader)
        print('success!')
    if now_task == 'climate_lstm':
        print(now_task)
        epoch = 50
        batch_size = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader, test_dataloader = data_process_climate(batch_size)
        lstm = LSTM(6, 288)
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, [20, 30], 0.1)
        writer = SummaryWriter('./result')
        train_climate(lstm, epoch, train_dataloader, criterion, optimizer, writer, test_dataloader)
        test(lstm, test_dataloader)
        print('success!')
