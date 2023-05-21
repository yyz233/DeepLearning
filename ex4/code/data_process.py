from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
import jieba
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from torch.nn.utils.rnn import pad_sequence

name2label_dict = defaultdict(int)
label2name_dict = defaultdict(str)


def my_collate(batch):
    """
    填充数据
    :param batch:
    :return:
    """
    inputs = [torch.as_tensor(data[0]) for data in batch]
    labels = [torch.as_tensor(data[1]) for data in batch]
    # 使用 pad_sequence 进行填充
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.as_tensor(labels)
    return [inputs, labels]


class OnlineShoppingDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return torch.tensor(np.array(self.data[item])), self.label[item]

    def __len__(self):
        return len(self.data)


class ClimateDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label


def return_online_shopping_dataloader(data, label, batch_size):
    dataset = OnlineShoppingDataset(data, label)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=my_collate
    )


def data_process_shopping(batch_size):
    shopping_path = './data/online_shopping_10_cats.csv'
    df = pd.read_csv(shopping_path)
    label = []
    data = []
    label_num = 0
    for index, row in df.iterrows():
        if row['cat'] not in name2label_dict:

            name2label_dict[row['cat']] = label_num
            label2name_dict[str(label_num)] = row['cat']
            label_num += 1
        label.append(name2label_dict[row['cat']])
        data.append(row['review'])
    # 以下转化词向量
    tokens = [jieba.lcut(i) for i in data]  # 分词
    model = Word2Vec(tokens, min_count=1, hs=1, window=3, vector_size=128)
    data_vector = [[model.wv[word] for word in sentence] for sentence in tokens]  # 转换成vector的reviews
    # 以下划分训练集、测试集、验证集
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    va_data = []
    va_label = []
    for i in range(len(data)):
        if i % 5 == 4:
            va_data.append(data_vector[i])
            va_label.append(label[i])
        elif i % 5 == 0:
            test_data.append(data_vector[i])
            test_label.append(label[i])
        else:
            train_data.append(data_vector[i])
            train_label.append(label[i])
    train_dataloader = return_online_shopping_dataloader(train_data, train_label, batch_size)
    test_dataloader = return_online_shopping_dataloader(test_data, test_label, batch_size)
    va_dataloader = return_online_shopping_dataloader(va_data, va_label, batch_size)
    return train_dataloader, test_dataloader, va_dataloader, label2name_dict
