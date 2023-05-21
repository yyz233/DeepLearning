import time
import torch


def validate(model, va_loader):
    model.eval()
    model.zero_grad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(va_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data).to(device)
        _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
        correct += (predicted == labels).sum().item()
        total += predicted.size(0)
    print('Accuracy on va dataset ' + str(correct / total))
    model.train()


def train(model, epoch, train_loader, criterion, optimizer, writer, name, va_loader):
    """
    for i, (data, labels) in enumerate(train_loader):
        print(data)
        print(data.shape)
        break
    return
    """
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    now_iter = 0
    for epoch in range(epoch):
        correct = 0
        total = 0
        total_loss = 0
        for i, (data, labels) in enumerate(train_loader):
            now_iter += 1
            data = data.to(device)
            labels = labels.to(device)
            # print(data.shape)
            # data_packed = nn.utils.rnn.pack_padded_sequence(data, lengths=data_lengths, batch_first=True, enforce_sorted=False)
            # print(data_packed)
            # break
            # 前向传播
            outputs = model(data).to(device)
            loss = criterion(outputs, labels).to(device)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计准确率和最新的loss
            _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
            correct += (predicted == labels).sum().item()
            total += predicted.size(0)
            total_loss += float(loss.item())
        now = time.time()
        writer.add_scalar(name + '_' + str(model) + '_loss', total_loss, epoch + 1)
        writer.add_scalar(name + '_' + str(model) + '_accuracy', correct / total, epoch + 1)
        print('epoch ' + str(epoch) + ': Accuracy on train dataset '
              + str(correct / total) + ' total loss: ' + str(total_loss))
        print('lasts:' + str(now - start) + ' s')
        if epoch % 9 == 0:
            validate(model, va_loader=va_loader)
            torch.save(model, './save_model/' + name + '_' + str(model) + '.ckpt')
    writer.close()


def test(model, test_loader):
    model.eval()
    model.zero_grad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data).to(device)
        _, predicted = torch.max(outputs, 1)  # max函数返回最大值和索引的元组，我们仅需用到索引值作为标签
        correct += (predicted == labels).sum().item()
        total += predicted.size(0)
    print('Accuracy on test dataset ' + str(correct / total))
