import random
import numpy as np
import torch
from utils.utils import standardization, get_feats_and_labels


def data_pro(args):
    # data_path = args.ae_data_path
    # data1 = torch.load(data_path).numpy()
    # data_path = args.oringin_data_path
    # data2 = np.loadtxt(data_path, delimiter=',', skiprows=1)
    # print(1)

    if args.use_AE:
        data_path = args.ae_data_path
        data = torch.load(data_path).numpy()
    else:
        data_path = args.oringin_data_path
        data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    bond_idx = list(set(data[:, 0]))
    num_row, num_col = data.shape
    random.shuffle(bond_idx)
    train_index = bond_idx[:len(bond_idx)*2//3]
    val_index = bond_idx[len(bond_idx)*2//3:]
    test_index = [2,   5,   13,  15,  29,
                    39,  40,  168, 171, 178,
                    195, 216, 226, 230, 233,
                    242, 244, 247, 249, 254,
                    261, 264, 270, 285, 294,
                    332, 343]

    train_bool_index = np.zeros(num_row, dtype=bool)
    val_bool_index = np.zeros(num_row, dtype=bool)
    test_bool_index = np.zeros(num_row, dtype=bool)
    for i in range(num_row):
        if int(data[i, 0]) in test_index:
            test_bool_index[i] = True
        elif data[i, 0] in val_index:
            val_bool_index[i] = True
        else:
            train_bool_index[i] = True

    train_data = data[train_bool_index]
    val_data = data[val_bool_index]
    test_data = data[test_bool_index]

    # 2. 对训练数据进行标准化
    if args.use_AE:
        except_col = [0]
    else:
        except_col = [0, 53]
    except_index = np.zeros(num_col, dtype=bool)
    except_index[except_col] = True
    mu, sigma, train_data[:, ~except_index] = standardization(train_data[:, ~except_index], None, None)
    _, _, val_data[:, ~except_index] = standardization(val_data[:, ~except_index], mu, sigma)
    _, _, test_data[:, ~except_index] = standardization(test_data[:, ~except_index], mu, sigma)
    

    # 3. 对序列进行划分，并得到特征和标签
    train_data = get_feats_and_labels(train_data, args.seq_length)
    val_data = get_feats_and_labels(val_data, args.seq_length)
    test_data = get_feats_and_labels(test_data, args.seq_length)

    return mu, sigma, train_data, val_data, test_data


# # cell test
# import sys
# sys.path.append('./')
# sys.path.append('../')
# from utils.args_lstm import args
# import numpy as np
# from utils.utils import standardization, get_feats_and_labels
# print(args)
# a = data_pro(args)
