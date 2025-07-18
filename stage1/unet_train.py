import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import unets

def train(net, train_dataloader, valid_dataloader, device, num_epoch, lr):
    print('training on:', device)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params_list['task_scheduler_step_size'], gamma=params_list['task_scheduler_gamma'])
    best_loss = 1000
    for epoch in range(num_epoch):
        time.sleep(0.03)
        net.train()

        train_loss = 0.0
        for data, label in tqdm(train_dataloader):
            data, label = data.to(device), label.to(device)
            predict = net(data)
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss = train_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Loss: {train_loss:.4f}')

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device), label.to(device)
                predict = net(data)
                loss = criterion(predict, label)
                test_loss += loss.item()

            test_loss = test_loss / len(valid_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epoch}],Test Loss: {test_loss:.4f}')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), '../save_model/segment_model.pth')
    print(f"best loss: {best_loss:.4f}")

    return net


if __name__ == '__main__':
    signals = torch.randn(10, 1, 1024)
    targets = torch.randn(10, 1, 1024)

    x_train, x_test, y_train, y_test = train_test_split(signals, targets, test_size=0.1, random_state=42)
    trainLoader = DataLoader(TensorDataset(x_train, y_train), batch_size=params_list['batch_size'], shuffle=True, drop_last=False)
    testLoader = DataLoader(TensorDataset(x_test, y_test), batch_size=params_list['batch_size'], shuffle=True, drop_last=False)

    segment_model = unets.Attn_UNet(img_ch=1, output_ch=1).cuda()
    segment_model = train(segment_model, trainLoader, testLoader, device=torch.device('cuda:0'), num_epoch=params_list['num_epoch'], lr=params_list['lr'])
