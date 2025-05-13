from diffusion import *
from torch.utils.data import DataLoader, TensorDataset

# ——————————————————————————————————————————————————————————————
if __name__ == '__main__':
    bcg = 0.1 * torch.randn(20, 1, 1024)
    ecg = 0.1 * torch.randn(20, 1, 1024)

    batch_size = 16
    ecg_data = torch.cat([ecg], dim=0)
    bcg_data = torch.cat([bcg], dim=0)

    train_loader = DataLoader(TensorDataset(ecg_data, bcg_data), batch_size=batch_size, shuffle=True, drop_last=False)
    model = VAR()
    model = train_Diff(
        model=model,
        train_loader=train_loader,
        num_epochs=params_list['task_epoch'],
        lr=params_list['task_lr'],
        device='cuda',
    )
