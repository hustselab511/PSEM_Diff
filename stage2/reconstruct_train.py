import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stage1 import unets
from stage2.rec_unets import AutoSemanticSegment

# ——————————————————————————————————————————————————————————————
# 参数设定
params_list = {}

params_list['num_epoch'] = 5
params_list['lr'] = 0.0001
params_list['batch_size'] = 32
params_list['task_scheduler_step_size'] = 5
params_list['task_scheduler_gamma'] = 0.8

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Conv1d:
        nn.init.xavier_normal_(m.weight)


def random_mask(bcg):
    tensor = bcg.clone()
    for i in range(32):
        random_x = random.randint(0, tensor.shape[-1] // 16 * 15)
        tensor[:, :, random_x:random_x + tensor.shape[-1] // 128] = 0
    return tensor


def make_energy_seq(sequence, window_size=21):  # N 1 length
    stride = 1
    unfolded = sequence.unfold(2, window_size, stride)  # N 1 length window_size
    energy_seq = torch.sum(unfolded * unfolded, dim=-1)
    return energy_seq


def cos_similarity_loss(tensor1, tensor2):
    tensor1 = tensor1.squeeze(1)
    tensor2 = tensor2.squeeze(1)
    tensor1 = F.normalize(tensor1)
    tensor2 = F.normalize(tensor2)
    similarity = torch.mm(tensor1, tensor2.transpose(-1, -2))
    loss = torch.sum(1 - similarity)
    return loss


def train(net, train_dataloader, valid_dataloader, device, num_epoch, lr):
    segment_model = unets.Attn_UNet(img_ch=1, output_ch=1).cuda()
    segment_model.load_state_dict(torch.load('../save_model/segment_model.pth'))
    segment_model.eval()

    print('training on:', device)
    net.to(device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params_list['task_scheduler_step_size'], gamma=params_list['task_scheduler_gamma'])
    best_loss = 100000
    for epoch in range(num_epoch):
        time.sleep(0.03)
        net.train()

        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss = 0.0
        for bcg, ecg in tqdm(train_dataloader):
            bcg, ecg = bcg.to(device), ecg.to(device)
            _, _, _, _, _, _, ecg_attn1, ecg_attn2, ecg_attn3, ecg_attn4 = segment_model.forward_attention(ecg)
            bcg_rec, bcg_feature, _, _, _, _, _, bcg_attn1, bcg_attn2, bcg_attn3, bcg_attn4 = net.get_features(bcg)
            loss1 = criterion1(bcg_rec, bcg)  # + criterion1(make_energy_seq(bcg_feature), make_energy_seq(ecg)) * 0.0001  # 添加能量一致性约束 损失项
            loss2 = criterion2(ecg_attn1, bcg_attn1) + criterion2(ecg_attn2, bcg_attn2) + criterion2(ecg_attn3, bcg_attn3) + criterion2(ecg_attn4, bcg_attn4)
            loss = loss1 + loss2 * 10
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss += loss.item()

        scheduler.step()
        train_loss = train_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Loss: {train_loss:.4f}')
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for bcg, ecg in valid_dataloader:
                bcg, ecg = bcg.to(device), ecg.to(device)
                _, _, _, _, _, _, ecg_attn1, ecg_attn2, ecg_attn3, ecg_attn4 = segment_model.forward_attention(ecg)
                bcg_rec, _, _, _, _, _, _, bcg_attn1, bcg_attn2, bcg_attn3, bcg_attn4 = net.get_features(bcg)
                loss1 = criterion1(bcg_rec, bcg)  # + criterion1(make_energy_seq(bcg_rec), make_energy_seq(ecg)) * 0.0001
                loss2 = criterion2(ecg_attn1, bcg_attn1) + criterion2(ecg_attn2, bcg_attn2) + criterion2(ecg_attn3, bcg_attn3) + criterion2(ecg_attn4, bcg_attn4)
                loss = loss1 + loss2 * 10
                test_loss += loss.item()

            test_loss = test_loss / len(valid_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epoch}],Test Loss: {test_loss:.4f}')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), '../save_model/reconstruct_model.pth')
            print("Update Save Model")
    print(f"best loss: {best_loss:.4f}")
    return net


if __name__ == '__main__':
    bcg = torch.randn(10, 1, 1024)
    ecg = torch.randn(10, 1, 1024)

    x_train, x_test, y_train, y_test = train_test_split(bcg, ecg, test_size=0.2, random_state=42)

    trainLoader = DataLoader(TensorDataset(x_train, y_train), batch_size=params_list['batch_size'], shuffle=True, drop_last=False)
    testLoader = DataLoader(TensorDataset(x_test, y_test), batch_size=params_list['batch_size'], shuffle=True, drop_last=False)

    reconstruct_model = AutoSemanticSegment().cuda()

    reconstruct_model = train(reconstruct_model, trainLoader, testLoader, device=torch.device('cuda:0'), num_epoch=params_list['num_epoch'], lr=params_list['lr'])
