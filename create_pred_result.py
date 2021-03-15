import torch
import torch.nn.functional as F
import time
import numpy as np
from utils import np_to_torch
import random
import pandas as pd

TRAIN_DATA_PATH = 'data/train_data_better_pred_v1.npy'
TEST_DATA_PATH = 'data/test_data_better_pred_v1.npy'
RESULT_SAVE_PATH = 'data/results/'


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
    train_xy_torch = np_to_torch(train_data)
    test_xy_torch = np_to_torch(test_data)
    xtrain = train_xy_torch[:, :, :-1]
    ytrain = torch.reshape(train_xy_torch[:, -1, -1], (train_xy_torch.shape[0], 1))
    xtest = test_xy_torch[:, :, :-1]
    ytest = torch.reshape(test_xy_torch[:, -1, -1], (test_xy_torch.shape[0], 1))
    return xtrain, ytrain, xtest, ytest


def train(xtrain, ytrain, hidden_d, layers, dropout, learning_rate, n_epoch, pic_name, batch_size, device):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(0)
    model = Model(input_dim=xtrain.shape[-1]-1, hidden_dim=hidden_d, n_layer=layers, drop_out=dropout).to(device)
    checkpoints = torch.load(RESULT_SAVE_PATH+'1_checkpoint.pth.tar')
    model.load_state_dict(checkpoints['state_dict'])
    return model


def run_results(model, xtrain, ytrain, xtest, ytest):
    ids = []
    ypreds = []
    ytrues = []
    allcids = list(set(xtrain.cpu().numpy()[:, -1, 0]))
    print(len(allcids))
    for cid in allcids:
        tmpx = xtrain[xtrain[:, -1, 0]==cid]
        tmpy = ytrain[xtrain[:, -1, 0]==cid]
        predy = model(tmpx[:, :, 1:]).cpu().detach().numpy()
        tmpx = tmpx.cpu().numpy()
        tmpy = tmpy.cpu().numpy()
        for i in range(tmpx.shape[0]):
            ids.append(tmpx[i, -1, 0])
            ypreds.append(predy[i, 0])
            ytrues.append(tmpy[0, 0])
        ids.append(ids[-1])
        ypreds.append(0)
        ytrues.append(ytrues[-1])
    allcids = list(set(xtest.cpu().numpy()[:, -1, 0]))
    print(len(allcids))
    for cid in allcids:
        tmpx = xtest[xtest[:, -1, 0]==cid]
        tmpy = ytest[xtest[:, -1, 0]==cid]
        predy = model(tmpx[:, :, 1:]).cpu().detach().numpy()
        tmpx = tmpx.cpu().numpy()
        tmpy = tmpy.cpu().numpy()
        for i in range(tmpx.shape[0]):
            ids.append(tmpx[i, -1, 0])
            ypreds.append(predy[i, 0])
            ytrues.append(tmpy[0, 0])
        ids.append(ids[-1])
        ypreds.append(0)
        ytrues.append(ytrues[-1])
    print(len(ids))
    df = pd.DataFrame({'CID':ids, 'ypred': ypreds, 'ytrue': ytrues})
    return df

def main(argv):
    start_time = time.time()
    xtrain, ytrain, xtest, ytest = load_data()
    print('Data loaded. Time spent is %f. ' % (time.time() - start_time))
    print('Xtrain data shape = ', xtrain.shape, ', ytrain data shape = ', ytrain.shape, '.')
    print('Xtest data shape = ', xtest.shape, ', ytest data shape = ', ytest.shape, '.\n')

    print('Begin training.')
    N_HIDDENS, N_LAYERS, DROPOUT, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, FLAG = \
        int(argv[0]), int(argv[1]), float(argv[2]), float(argv[3]), int(argv[4]), int(argv[5]), argv[6]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is %s.' % device)
    xtrain = xtrain.to(device)
    ytrain = ytrain.to(device)
    xtest = xtest.to(device)
    ytest = ytest.to(device)
    model = train(xtrain, ytrain, hidden_d=N_HIDDENS, layers=N_LAYERS,
                  dropout=DROPOUT, learning_rate=LEARNING_RATE, n_epoch=N_EPOCHS, pic_name=FLAG,
                  batch_size=BATCH_SIZE, device=device)
    print('>>> Begin measuring model.')
    df = run_results(model, xtrain, ytrain, xtest, ytest)
    df.to_csv('pred_result.csv', index=False)


if __name__ == '__main__':
    argv = ['512', '1', '0.0', '0.0003', '500', '2048', '1']
    main(argv)
