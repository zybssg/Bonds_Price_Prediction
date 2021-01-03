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

# 读取文件
result_path = './output/result.pt'
result = torch.load(result_path)
mu, sigma, test_data = result['standardization_params']
test_feats = torch.from_numpy(test_data['feats'])
test_labels = torch.from_numpy(test_data['labels'])

model = result['model']
loss_func = nn.MSELoss()
# model.load_state_dict(result['model'])

model.eval()
test_preds = model(test_feats)
test_loss = loss_func(test_preds, test_labels.unsqueeze(-1))

plt.plot(test_preds.squeeze(-1).detach().numpy()[:3000], label='Prediction')
plt.plot(test_labels.squeeze(-1).numpy()[:3000], label='Ground Truth')
plt.legend()
plt.savefig('./output/pred.jpg')
