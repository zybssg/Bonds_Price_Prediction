import argparse


parse = argparse.ArgumentParser(description="the super params of the LSTM model")

# train
parse.add_argument('--cur_state', type=str, default='train', \
    choices=['train', 'validation', 'test'], dest='the current state')
parse.add_argument('--epochs', type=int, default=50)
parse.add_argument('--batch_size', type=int, default=128)
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--show_epoch', type=int, default=2)

# model
parse.add_argument('--use_AE', type=bool, default=False)
parse.add_argument('--input_dim', type=int)
parse.add_argument('--fc1_dim', type=int, default=32)
parse.add_argument('--lstm_dim', type=int, default=64)
parse.add_argument('--fc2_dim', type=int, default=16)
parse.add_argument('--output_dim', type=int, default=1)
parse.add_argument('--num_lstm', type=int, default=1)

# data
parse.add_argument('--oringin_data_path', type=str, default='./data/data_55.csv')
parse.add_argument('--ae_data_path', type=str, default='./data/data_11.csv')
parse.add_argument('--seq_length', type=int, default=5)
parse.add_argument('--save_path', type=str, default='./output/result.pt')

args = parse.parse_args()
if args.use_AE:
    args.input_dim = 10
else:
    args.input_dim = 59
