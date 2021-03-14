import torch
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from pytorchtools import EarlyStopping
import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from utils import np_to_torch
import random
import matplotlib.pyplot as plt
import os

TRAIN_DATA_PATH = 'BPIC2012/train_data_set.npy'
TEST_DATA_PATH = 'BPIC2012/test_data_set.npy'
RESULT_SAVE_PATH = 'BPIC2012/results/'


class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, drop_out):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layer, batch_first=True,
                                  dropout=drop_out)
        self.dense = torch.nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = F.relu(self.dense(out))
        return out


def load_data():
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    train_xy, valid_xy = train_test_split(train_data, test_size=0.2, shuffle=True, random_state=0)
    train_xy_torch = np_to_torch(train_xy)
    valid_xy_torch = np_to_torch(valid_xy)
    test_xy_torch = np_to_torch(test_data)
    xtrain = train_xy_torch[:, :, :-1]
    ytrain = torch.reshape(train_xy_torch[:, -1, -1], (train_xy_torch.shape[0], 1))
    xvalid = valid_xy_torch[:, :, :-1]
    yvalid = torch.reshape(valid_xy_torch[:, -1, -1], (valid_xy_torch.shape[0], 1))
    xtest = test_xy_torch[:, :, :-1]
    ytest = torch.reshape(test_xy_torch[:, -1, -1], (test_xy_torch.shape[0], 1))
    return xtrain, ytrain, xvalid, yvalid, xtest, ytest


def train(xtrain, ytrain, xvalid, yvalid, hidden_d, layers, dropout, learning_rate, n_epoch, pic_name, batch_size, device):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def plot_loss(train_loss, valid_loss):
        plt.figure(figsize=(20, 10))
        plt.plot(train_loss, 'b', label='train_loss')
        plt.plot(valid_loss, 'r', label='valid_loss')
        plt.legend()
        # plt.show()
        plt.savefig(RESULT_SAVE_PATH + pic_name + '.jpg')

    train_dataset = TensorDataset(xtrain, ytrain)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    setup_seed(0)
    model = Model(input_dim=xtrain.shape[-1], hidden_dim=hidden_d, n_layer=layers, drop_out=dropout).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, eps=1e-4)
    early_stopping = EarlyStopping(patience=50, verbose=True)

    train_loss = []
    valid_loss = []
    for epoch in range(n_epoch):
        train_loss_tmp = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_tmp += loss.data
        train_loss.append(train_loss_tmp / (step + 1))

        model.eval()
        valid_output = model(xvalid)
        valid_loss_data = criterion(valid_output, yvalid)
        scheduler.step(valid_loss_data)
        valid_loss.append(valid_loss_data.data)
        print(
            'EPOCH: %d, TRAINING LOSS: %f, VALIDATION LOSS: %f' % (epoch, train_loss_tmp / (step + 1), valid_loss_data))
        early_stopping(valid_loss_data, model)
        if early_stopping.early_stop:
            print('Early stopped.')
            break
        model.train()
    plot_loss(train_loss, valid_loss)
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model


def model_metrics(model, x, ytrue, device):
    if device == 'cpu':
        tmp = model(x).detach().numpy()
    else:
        tmp = model(x).cpu().detach().numpy()
        ytrue = ytrue.cpu()
    predy = np.reshape(tmp, (tmp.shape[0],))
    result = 'MSE: %f, MAE: %f.' % (mean_squared_error(y_true=ytrue, y_pred=predy),
                                    mean_absolute_error(y_true=ytrue, y_pred=predy))
    print(result)
    return result


def write_results(argv, train_info, valid_info, test_info):
    N_HIDDENS, N_LAYERS, DROPOUT, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, FLAG = \
        argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]
    f = open(RESULT_SAVE_PATH + FLAG + '.txt', 'w', encoding='utf-8')
    f.write('n_hidden=%s, n_layers=%s, dropout=%s, learning_rate=%s, epochs=%s, batch_size=%s.\n' %
            (N_HIDDENS, N_LAYERS, DROPOUT, LEARNING_RATE, N_EPOCHS, BATCH_SIZE))
    f.write('\nTraining Result:\n')
    f.write(train_info + '\n')
    f.write('\nValidation Result:\n')
    f.write(valid_info + '\n')
    f.write('\nTest Result:\n')
    f.write(test_info + '\n')
    f.close()


def main(argv):
    start_time = time.time()
    xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data()
    print('Data loaded. Time spent is %f. ' % (time.time() - start_time))
    print('Xtrain data shape = ', xtrain.shape, ', ytrain data shape = ', ytrain.shape, '.')
    print('Xvalidation data shape = ', xvalid.shape, ', yvalidation data shape = ', yvalid.shape, '.')
    print('Xtest data shape = ', xtest.shape, ', ytest data shape = ', ytest.shape, '.\n')

    print('Begin training.')
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)
    N_HIDDENS, N_LAYERS, DROPOUT, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, FLAG = \
        int(argv[0]), int(argv[1]), float(argv[2]), float(argv[3]), int(argv[4]), int(argv[5]), argv[6]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is %s.' % device)
    xtrain = xtrain.to(device)
    ytrain = ytrain.to(device)
    xvalid = xvalid.to(device)
    yvalid = yvalid.to(device)
    xtest = xtest.to(device)
    ytest = ytest.to(device)
    model = train(xtrain, ytrain, xvalid, yvalid, hidden_d=N_HIDDENS, layers=N_LAYERS,
                  dropout=DROPOUT, learning_rate=LEARNING_RATE, n_epoch=N_EPOCHS, pic_name=FLAG,
                  batch_size=BATCH_SIZE, device=device)
    print('Finished model training. Time is %f.' % (time.time() - start_time))
    torch.save({'state_dict': model.state_dict()}, RESULT_SAVE_PATH + FLAG + '_checkpoint.pth.tar')
    print('Model trained. Time is %f.\n' % (time.time() - start_time))

    print('>>> Begin measuring model.')
    train_info = model_metrics(model, xtrain, ytrain, device)
    valid_info = model_metrics(model, xvalid, yvalid, device)
    test_info = model_metrics(model, xtest, ytest, device)
    write_results(argv, train_info, valid_info, test_info)
    print('Ended. Time is %f.' % (time.time() - start_time))


if __name__ == '__main__':
    # argv = ['2', '1', '0.01', '5', '2048', '0']
    main(sys.argv[1:])
    # main(argv)
