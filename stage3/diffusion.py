import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from stage2 import rec_unets

# random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ——————————————————————————————————————————————————————————————
params_list = {}
params_list['task_train'] = True
params_list['task_epoch'] = 5
params_list['task_lr'] = 0.001
params_list['task_scheduler_step_size'] = 5
params_list['task_scheduler_gamma'] = 0.7
params_list['var_model_save_path'] = '../save_model/PSEM_diff_model.pth'


def flatten_data(data):
    return data.reshape(data.shape[0] * data.shape[1], data.shape[-1])


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        ans = nn.functional.layer_norm(data, (data.shape[-1],)).to(device=data.device)
        return ans


class VAR_BCG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        self.encoder = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.layer_norm = LayerNorm()

    def forward(self, data):
        data = self.encoder(data)
        data = self.layer_norm(data)
        return data


class VAR_BCG_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        self.decoder1 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.decoder2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.decoder3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.layer_norm = LayerNorm()

    def forward(self, data1, data2, data3):
        data1 = self.decoder1(data1)
        data2 = self.decoder2(data2)
        data3 = self.decoder3(data3)

        data3 = self.bilinear_interpolate(data3, data1)
        data2 = self.bilinear_interpolate(data2, data1)
        data1 = self.bilinear_interpolate(data1, data1)

        data_sum = (data1 + data2 + data3)
        data = self.layer_norm(data_sum)
        return data

    def bilinear_interpolate(self, data, target):
        target_length = target.shape[-1]
        data = data.unsqueeze(-2)
        output_tensor = F.interpolate(data, size=(1, target_length), mode='bilinear', align_corners=True)
        return output_tensor.squeeze(-2)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x


class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.feat = nn.Sequential(
            Bottleneck(channels, channels, reduction=2),
            nn.Conv1d(channels, 2, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        x = self.feat(x)
        x = x.transpose(-1, -2)
        x = x.softmax(2)[:, :, 0].squeeze(-1)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=21):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return self.residual_block(x) + self.shortcut(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upSampleSize=2, upSampleStride=2):
        super().__init__()
        kernel_size = 21
        self.residual_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upSampleSize, stride=upSampleStride),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        )
        self.shortcut = nn.Sequential()
        if upSampleSize != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upSampleSize, stride=upSampleStride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return self.residual_block(x) + self.shortcut(x)


class ResUnet(nn.Module):
    def __init__(self, channels_in, kernel_size):
        super().__init__()

        self.Encoder1 = EncoderBlock(in_channels=channels_in, out_channels=64, kernel_size=kernel_size)
        self.Encoder2 = EncoderBlock(in_channels=64, out_channels=128, kernel_size=kernel_size)
        self.Encoder3 = EncoderBlock(in_channels=128, out_channels=256, kernel_size=kernel_size)
        self.BottleEncoder = EncoderBlock(in_channels=256, out_channels=512, kernel_size=kernel_size)

        self.downSample = nn.MaxPool1d(kernel_size=2)

        self.upSampleBlock3 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.Decoder3 = EncoderBlock(in_channels=512, out_channels=256, kernel_size=kernel_size)

        self.upSampleBlock2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Decoder2 = EncoderBlock(in_channels=256, out_channels=128, kernel_size=kernel_size)

        self.upSampleBlock1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.Decoder1 = EncoderBlock(in_channels=128, out_channels=64, kernel_size=kernel_size)

        self.toOutput = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=channels_in, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels_in),
        )

    def forward(self, x):
        down1_conn = self.Encoder1(x)
        down1 = self.downSample(down1_conn)

        down2_conn = self.Encoder2(down1)
        down2 = self.downSample(down2_conn)

        down3_conn = self.Encoder3(down2)
        down3 = self.downSample(down3_conn)

        mid_out = self.BottleEncoder(down3)

        up4 = self.Decoder3(torch.cat([down3_conn, self.upSampleBlock3(mid_out)], dim=1))
        up3 = self.Decoder2(torch.cat([down2_conn, self.upSampleBlock2(up4)], dim=1))
        up2 = self.Decoder1(torch.cat([down1_conn, self.upSampleBlock1(up3)], dim=1))
        reConstruct = self.toOutput(up2)
        return reConstruct


class DiffusionScheduler(nn.Module):
    def __init__(self, beta_start=0.001, beta_end=0.02, train_time_steps_num=100, num_inference_steps=10):
        super().__init__()
        self.train_time_steps_num = train_time_steps_num
        self.num_inference_steps = num_inference_steps
        self.betas = torch.linspace(beta_start, beta_end, train_time_steps_num)
        self.alphas = 1 - self.betas
        self.alphas_list = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_value = torch.tensor(1.0)
        self.train_time_steps = torch.from_numpy(np.arange(0, train_time_steps_num)[::-1].copy().astype(np.int64)).int()

    def to_device(self, sample):
        self.betas = self.betas.to(sample.device)
        self.alphas_list = self.alphas_list.to(sample.device)
        self.final_alpha_value = self.final_alpha_value.to(sample.device)
        self.train_time_steps = self.train_time_steps.to(sample.device)
        return

    def add_noise(self, x_0, noise, time_step):
        self.to_device(x_0)
        alpha = self.alphas_list[time_step].unsqueeze(-1).unsqueeze(-1)
        beta = 1 - alpha
        noised_sample = (alpha ** 0.5) * x_0 + (beta ** 0.5) * noise
        return noised_sample

    def denoised(self, sample, noise_pred, cur_time_step):
        self.to_device(sample)
        prev_time_step = cur_time_step - self.train_time_steps_num // self.num_inference_steps
        cur_alpha = self.alphas_list[cur_time_step]
        prev_alpha = self.alphas_list[prev_time_step] if prev_time_step >= 0 else self.final_alpha_value
        cur_beta = 1 - cur_alpha
        prev_beta = 1 - prev_alpha
        x_0 = (sample - cur_beta ** 0.5 * noise_pred) / (cur_alpha ** 0.5)
        prev_sample = (prev_alpha ** 0.5) * x_0 + (prev_beta ** 0.5) * noise_pred
        return prev_sample

    def forward(self):
        print(self.train_time_steps)
        return


class DiffusionModel(nn.Module):
    def __init__(self, channels_in, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(2560, channels_in)
        self.pred = ResUnet(channels_in=channels_in, kernel_size=3)

    def forward(self, noisy_image, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image
        time_embedding = self.time_embedding(t)[..., None]
        feat = feat + time_embedding
        ret = self.pred(feat)
        return ret


class VAR_BCG_Down_Sampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.mid_channels = 32
        self.kernel_size = 21

        self.down_sampling2 = self.down_sampling_layer(stride=2 ** 3)
        self.down_sampling1 = self.down_sampling_layer(stride=2 ** 1)

    def down_sampling_layer(self, stride):
        return nn.Sequential(
            nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=stride * 2 + 1, padding=stride, bias=False),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=stride * 2 + 1, padding=stride, bias=False),
            nn.BatchNorm1d(self.mid_channels),
            nn.AvgPool1d(kernel_size=stride * 2 + 1, stride=stride, padding=stride),
        )

    def bilinear_interpolate(self, data, target):
        target_length = target.shape[-1]
        data = data.unsqueeze(-2)
        output_tensor = F.interpolate(data, size=(1, target_length), mode='bilinear', align_corners=True)
        return output_tensor.squeeze(-2)

    def forward(self, bcg):
        down2 = self.down_sampling2(bcg)
        bcg_res = bcg - self.bilinear_interpolate(down2, bcg)
        down1 = self.down_sampling1(bcg_res)
        bcg_res = bcg_res - self.bilinear_interpolate(down1, bcg_res)

        return [down2, down1, bcg_res]


class VAR_Diffusion(nn.Module):
    def __init__(self, train_noise_param=0.1, valid_noise_param=0.001):
        super().__init__()
        self.train_noise_param = train_noise_param
        self.valid_noise_param = valid_noise_param
        self.mid_channels = 32
        self.T = 1000
        self.inference_steps = 200
        self.noise_adapter = NoiseAdapter(channels=self.mid_channels, kernel_size=3)
        self.diffusion_scheduler = DiffusionScheduler(beta_start=0.001, beta_end=0.002, train_time_steps_num=self.T, num_inference_steps=self.inference_steps)
        self.diffusion = DiffusionModel(channels_in=self.mid_channels, kernel_size=3)
        self.mse_loss = nn.MSELoss()
        self.layer_norm = LayerNorm()

    def forward(self, bcg_feature):
        denoised_bcg_feature = self.denoised_valid(bcg_feature)
        return denoised_bcg_feature

    def denoised_valid(self, student):
        adapt_noise = self.noise_adapter(student).unsqueeze(-2)
        noisy_student = torch.sqrt(adapt_noise) * student + self.valid_noise_param * torch.sqrt(1 - adapt_noise) * torch.randn_like(student).to(student.device)
        for cur_time_step in self.diffusion_scheduler.train_time_steps[::self.inference_steps // 2]:
            noise_predict = self.diffusion(noisy_student, torch.tensor(cur_time_step).to(student.device))
            noisy_student = self.diffusion_scheduler.denoised(noisy_student, noise_predict, cur_time_step)
            noisy_student = self.layer_norm(noisy_student)
        denoised_student = noisy_student
        return denoised_student

    def denoised_train(self, student):
        adapt_noise = self.noise_adapter(student).unsqueeze(-2)
        noisy_student = torch.sqrt(adapt_noise) * student + self.train_noise_param * torch.sqrt(1 - adapt_noise) * torch.randn_like(student).to(student.device)
        for cur_time_step in self.diffusion_scheduler.train_time_steps[::self.inference_steps // 2]:
            noise_predict = self.diffusion(noisy_student, torch.tensor(cur_time_step).to(student.device))
            noisy_student = self.diffusion_scheduler.denoised(noisy_student, noise_predict, cur_time_step)
            noisy_student = self.layer_norm(noisy_student)
        denoised_student = noisy_student
        return denoised_student

    def get_loss1(self, teacher, student):
        denoised_student = self.denoised_train(student)
        loss1 = self.mse_loss(teacher, denoised_student)
        return loss1

    def get_loss2(self, teacher):
        noise = torch.randn_like(teacher).to(teacher.device)
        time_step = torch.randint(0, self.T // 10, (teacher.shape[0],)).long().to(teacher.device)
        noisy_teacher = self.diffusion_scheduler.add_noise(teacher, noise, time_step)
        noise_predict = self.diffusion(noisy_teacher, time_step)
        loss2 = self.mse_loss(noise, noise_predict)
        return loss2


class VAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.var_bcg_encoder = VAR_BCG_Encoder()
        self.var_bcg_down_sampling = VAR_BCG_Down_Sampling()
        self.var_ecg_down_sampling = VAR_BCG_Down_Sampling()
        self.var_ecg_encoder = VAR_BCG_Encoder()
        self.var_ecg_decoder = VAR_BCG_Decoder()
        self.var_bcg_diffusion1 = VAR_Diffusion(train_noise_param=0.1, valid_noise_param=0.001)
        self.var_bcg_diffusion2 = VAR_Diffusion(train_noise_param=0.1, valid_noise_param=0.001)
        self.var_bcg_diffusion3 = VAR_Diffusion(train_noise_param=0.1, valid_noise_param=0.001)
        self.mse_loss = nn.MSELoss()

        self.rec_model = rec_unets.AutoSemanticSegment()
        self.rec_model.load_state_dict(torch.load('../save_model/reconstruct_model.pth'))
        self.rec_model.eval()

    def bilinear_interpolate(self, data, target):
        target_length = target.shape[-1]
        data = data.unsqueeze(-2)
        output_tensor = F.interpolate(data, size=(1, target_length), mode='bilinear', align_corners=True)
        return output_tensor.squeeze(-2)

    def get_var_loss(self, bcg, ecg):
        bcg_rec, bcg_feature, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4 = self.rec_model.get_features(bcg)
        bcg_feature = self.var_bcg_encoder(bcg_feature)
        bcg_down_sampling_list = self.var_bcg_down_sampling(bcg_feature)
        interpolate_bcg_down3 = self.var_bcg_diffusion3.denoised_train(bcg_down_sampling_list[0])  # low
        interpolate_bcg_down2 = self.var_bcg_diffusion2.denoised_train(bcg_down_sampling_list[1])  # mid
        interpolate_bcg_down1 = self.var_bcg_diffusion1.denoised_train(bcg_down_sampling_list[2])  # high
        ecg_pred = self.var_ecg_decoder(interpolate_bcg_down1, interpolate_bcg_down2, interpolate_bcg_down3)
        loss2 = self.mse_loss(ecg_pred, ecg)
        return loss2 * 20

    def forward(self, bcg):
        bcg_rec, bcg_feature, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4 = self.rec_model.get_features(bcg)
        bcg_feature = self.var_bcg_encoder(bcg_feature)
        bcg_down_sampling_list = self.var_bcg_down_sampling(bcg_feature)
        interpolate_bcg_down3 = self.var_bcg_diffusion3.denoised_valid(bcg_down_sampling_list[0])  # low
        interpolate_bcg_down2 = self.var_bcg_diffusion2.denoised_valid(bcg_down_sampling_list[1])  # mid
        interpolate_bcg_down1 = self.var_bcg_diffusion1.denoised_valid(bcg_down_sampling_list[2])  # high
        ecg_pred = self.var_ecg_decoder(interpolate_bcg_down1, interpolate_bcg_down2, interpolate_bcg_down3)
        return ecg_pred


def train_Diff(*, model, train_loader, num_epochs=5, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params_list['task_scheduler_step_size'], gamma=params_list['task_scheduler_gamma'])
    best_loss = 999999.0
    for epoch in range(num_epochs):
        print(f'------------- VAR Epoch {epoch + 1}/{num_epochs} -------------')
        model = model.to(device).train()
        total_loss = 0
        batches = 0
        for (batch_ecg_sample, batch_bcg_sample) in tqdm(train_loader, desc='Stage2 (2/2)'):
            batch_ecg_sample = batch_ecg_sample.to(device)
            batch_bcg_sample = batch_bcg_sample.to(device)

            loss = model.get_var_loss(batch_bcg_sample, batch_ecg_sample)
            total_loss += loss.item()
            batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        print(f'Total_Loss: {total_loss:.6f}, Avg_Batch_Loss: {total_loss / batches:.6f}')

        if total_loss / batches <= best_loss:
            torch.save(model.cpu().eval(), params_list['var_model_save_path'])
            best_loss = total_loss / batches
    return model.cpu()
