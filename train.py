import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.DoubleTensor)

from data.data import Dataset_Bonds
from data.data_process import data_pro
from models.model import Bonds_LSTM
from utils.args_lstm import args

# 1. 处理data
mu, sigma, train_data, val_data, test_data = data_pro(args)

# 2. 构建dataset和dataloader
train_dataset = Dataset_Bonds(train_data)
# val_dataset = Dataset_Bonds(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# val_dataset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

# 3. 训练初始化
model = Bonds_LSTM(args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_func = nn.MSELoss()
loss_train_all = np.zeros((args.epochs+1))
loss_val_all = np.zeros((args.epochs+1))

# 4. 计算初始时刻的train loss和validation loss
train_feats = torch.from_numpy(train_data['feats'])
train_labels = torch.from_numpy(train_data['labels'])
train_preds = model(train_feats)
train_loss = loss_func(train_preds, train_labels.unsqueeze(-1))
loss_train_all[0] = train_loss

val_feats = torch.from_numpy(val_data['feats'])
val_labels = torch.from_numpy(val_data['labels'])
val_preds = model(val_feats)
val_loss = loss_func(val_preds, val_labels.unsqueeze(-1))
loss_val_all[0] = val_loss
print('epoch=%d || training_loss=%.4f || val_loss=%.4f'%(0, train_loss, val_loss))

# 5. training
for epoch in range(args.epochs):
    train_loss_sum = 0
    for batch_idx, batch_train_data in enumerate(train_dataloader):
        model.train()
        batch_train_feats, batch_train_labels = batch_train_data
        batch_train_preds = model(batch_train_feats)
        train_loss = loss_func(batch_train_preds, batch_train_labels.unsqueeze(-1))
        train_loss_sum += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    train_loss = train_loss_sum/(batch_idx+1)
    loss_train_all[epoch+1] = train_loss.item()

    # 计算验证集误差
    model.eval()
    val_preds = model(val_feats)
    val_loss = loss_func(val_preds, val_labels.unsqueeze(-1))
    loss_val_all[epoch+1] = val_loss.item()
    if epoch % args.show_epoch == 0 and epoch != 0:
        print('epoch=%d || training_loss=%.4f || val_loss=%.4f'%(epoch+1, train_loss, val_loss))

# 保存结果
save_data = {
             'standardization_params': (mu, sigma, test_data),
             'loss_train_all': loss_train_all,
             'loss_val_all': loss_val_all,
             'model': model
            }
torch.save(save_data, os.path.join(args.save_path, 'result_%d.pt'%args.input_dim))


# 1. 早停
# 2. cuda
