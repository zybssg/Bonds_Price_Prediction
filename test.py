import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

from data.data import Dataset_Bonds
from data.data_process import data_pro
from models.model import Bonds_LSTM
from utils.args_lstm import args
from utils.utils import re_standardization

args.input_dim = 59 # chose from [1, 8, 59]
print('args.input_dim = %d'%args.input_dim)
# 读取文件
result_path = './output/result_%d.pt'%args.input_dim
result = torch.load(result_path)
mu, sigma, test_data = result['standardization_params']
test_feats = torch.from_numpy(test_data['feats'])
test_labels = torch.from_numpy(test_data['labels'])
test_bond_index = torch.from_numpy(test_data['bond_idx'])
loss_train_all = result['loss_train_all']
loss_val_all = result['loss_val_all']
# loss_val_all[0] = 0.8
# loss_val_all[1] = 0.4
# loss_val_all[2] = 0.3


model = result['model']
loss_func = nn.MSELoss()
# model.load_state_dict(result['model'])

model.eval()
test_preds = model(test_feats)
test_loss = loss_func(test_preds, test_labels.unsqueeze(-1))

test_preds = test_preds.squeeze(-1).detach()
test_labels = test_labels.squeeze(-1)


# 某支债券价格30天走势图
plt.figure(1)
idx = 13
pred_idx = test_preds[test_bond_index==idx]
labels_idx = test_labels[test_bond_index==idx]
# pred_30 = pred_idx[-31:-1]
# real_30 = labels_idx[-31:-1]
pred_30 = pred_idx[-100:]
real_30 = labels_idx[-100:]
mu_, sigmma_ = mu[-1], sigma[-1]

real_30_re = re_standardization(real_30, mu_, sigmma_)
pred_30_re = re_standardization(pred_30, mu_, sigmma_)

plt.plot(pred_30_re, label='Prediction')
plt.plot(real_30_re, label='Ground Truth')
plt.legend()
plt.savefig('./output/pred_%d.jpg'%args.input_dim, dpi=300)

# 训练误差
epochs = range(args.epochs + 1)
plt.figure(2)
plt.plot(epochs, loss_train_all)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('./output/train_loss_%d.jpg'%args.input_dim, dpi=300)

# 验证误差
plt.figure(3)
plt.plot(epochs, loss_val_all)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.savefig('./output/val_loss_%d.jpg'%args.input_dim, dpi=300)

# 计算指标
test_idx = [2,   5,   13,  15,  29,
            39,  40,  168, 171, 178,
            195, 216, 226, 230, 233,
            242, 244, 247, 249, 254,
            261, 264, 270, 285, 294,
            332, 343]
result_27bonds_30day = torch.zeros(27,30,2)
for i in range(len(test_idx)):
    idx = test_idx[i]
    result_27bonds_30day[i,:,0] = test_preds[test_bond_index==idx][-31:-1]
    result_27bonds_30day[i,:,1] = test_labels[test_bond_index==idx][-31:-1]
result_27bonds_30day = re_standardization(result_27bonds_30day, mu_, sigmma_)
result_27bonds_30day_810 = result_27bonds_30day.reshape(-1, 2)

diff = result_27bonds_30day_810[:,0] - result_27bonds_30day_810[:,1]
# MSE
MSE = np.mean((diff*diff).numpy())
# MAE
MAE = np.mean(np.abs(diff).numpy())

# 平均绝对偏离比例
result_27bonds_30day = result_27bonds_30day.permute(1,0,2)
ANS = np.mean((np.abs(result_27bonds_30day[:,:,0]-result_27bonds_30day[:,:,1])/result_27bonds_30day[:,:,1]).numpy(), 1)*100

print('MSE = %.4f, MAE = %.4f'%(MSE, MAE))
print("平均绝对偏离比例:")
print(ANS)