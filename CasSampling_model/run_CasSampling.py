import numpy as np
import pickle
import config
from torch.utils.data import dataset
from torch.utils.data import DataLoader
import torch

import CasSampling_model
import random
from tqdm.auto import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('Model_size：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


class CasData(dataset.Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.id_train, self.x, self.L, self.y_train, self.sz_train, self.time_train, self.interval_popularity_train,self.vocabulary_size = pickle.load(open(self.root_dir, 'rb'))
        self.n_time_interval = config.n_time_interval
        self.n_steps = config.n_steps
    def __getitem__(self, idx):
        id = self.id_train[idx]

        y = self.y_train[idx]

        L = self.L[idx].todense()
        interval_popuplarity = self.interval_popularity_train[idx]
        time = self.time_train[idx]
        time = np.array(time,dtype=float)

        # temp = np.zeros(100)
        # for i in range(len(time)):
        #     temp[i] = time[i]
        # time = temp


        x = self.x[idx]
        x_ = self.x[idx].todense()
        x_ = torch.tensor(x_,dtype=torch.float32)
        time = torch.tensor(time,dtype=torch.float32)
        L = torch.tensor(L, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # time_interval_index_sample = torch.tensor(time_interval_index_sample, dtype=torch.float32)
        # rnn_index_temp = torch.tensor(rnn_index_temp, dtype=torch.float32)
        size = self.sz_train[idx]
        size = np.log2(size)
        size_train = torch.tensor(size,dtype=torch.float32)
        interval_popuplarity = torch.tensor(interval_popuplarity,dtype=torch.float32)

        return x_, L, time, y,size_train,interval_popuplarity
        # return x_,L,time,y,time_interval_index_sample,rnn_index_temp,

    def __len__(self):
        return len(self.sz_train)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    #导入数据
    ############################################
    Dataset = CasData(config.train_data)
    setup_seed(args.seed)
    learning_rate = args.learning_rate
    weight_decay = 5e-4
    # train_dataset[5]
    # test_dataset = CasData(config.test_data)
    # val_dataset = CasData(config.val_data)
    test_size = int(len(Dataset)*0.15)
    val_size = int(len(Dataset)*0.15)
    train_size = int(len(Dataset)) - test_size - val_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(Dataset,
                                                                                  [train_size, test_size,
                                                                                   val_size],generator=torch.Generator().manual_seed(0))


    print("train_size:",train_size,"test_size:",test_size,"val_size:",val_size)
    train_dataloader = DataLoader(train_dataset,config.batch_size,drop_last=True)
    test_dataloader = DataLoader(test_dataset, config.batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, config.batch_size, drop_last=True)

    ############################################

    GCN_hidden_size=args.GCN_hidden_size
    GCN_hidden_size2 =args.GCN_hidden_size2
    MLP_hidden1 = args.MLP_hidden1
    MLP_hidden2 = args.MLP_hidden2
    Activation_fc = args.Activation_fc

    model = CasSampling_model.CasSamplingNet(input_dim=128,GCN_hidden_size=GCN_hidden_size,GCN_hidden_size2=GCN_hidden_size2,MLP_hidden1=MLP_hidden1,MLP_hidden2=MLP_hidden2,Activation_fc=Activation_fc)
    model = model.to(device)
    getModelSize(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    best_val_loss = 999999
    n_epochs = 50
    for epoch in range(1,n_epochs+1):
        print("-------------Epoch:{}-----------".format(epoch))
        train_loss = []
        train_MAPE = []

        test_loss = []
        test_MAPE = []
        val_loss = []
        val_MAPE = []


        for data in train_dataloader:
            x, L,time,y,size_train,interval_popularity= data
            time = time.to(device)
            x = x.to(device)
            L = L.to(device)

            y = y.to(device)
            y = torch.reshape(y, [config.batch_size, 1])
            interval_popularity = interval_popularity.to(device)

            pred,nodes_pred = model(L, x,time,interval_popularity)
            optimizer.zero_grad()
            loss = torch.mean(torch.pow((pred - y), 2))

            error = torch.mean(torch.pow((pred - y), 2))
            MAPE_error = torch.mean(torch.abs(pred-y)/(torch.log2(torch.pow(2,y)+1)+torch.log2(torch.pow(2,pred)+1))*2)

            loss.backward()
            optimizer.step()

            train_loss.append(error.item())
            train_MAPE.append(MAPE_error.item())


        print("Train_MSLE：{:.3f},Train_MAPE:{:.3f}".format(np.mean(train_loss),np.mean(train_MAPE)))
        torch.cuda.empty_cache()
        with torch.no_grad():
            for data in val_dataloader:
                x, L,time, y,sz,interval_popularity= data

                time = time.to(device)
                x = x.to(device)
                L = L.to(device)

                y = y.to(device)
                y = torch.reshape(y, [config.batch_size, 1])
                interval_popularity = interval_popularity.to(device)

                pred,_ = model(L,x,time,interval_popularity)
                loss = torch.mean(torch.pow((pred - y), 2))
                MAPE_error = torch.mean(torch.abs(pred-y)/(torch.log2(torch.pow(2,y)+1)+torch.log2(torch.pow(2,pred)+1))*2)
                val_loss.append(loss.item())
                val_MAPE.append(MAPE_error.item())

        print("Val_MSLE:{:.3f},Val_MAPE:{:.3f}".format(np.mean(val_loss),np.mean(val_MAPE)))
        if (best_val_loss > np.mean(val_loss)):
            best_val_loss = np.mean(val_loss)
            print("Save best model")
            torch.save(model.state_dict(), 'model.pth')

    torch.cuda.empty_cache()
    model.load_state_dict(torch.load('./model.pth'))
    with torch.no_grad():
        for data in test_dataloader:
            x,L,time,y,sz,interval_popularity= data

            time = time.to(device)
            x = x.to(device)
            L = L.to(device)

            y = y.to(device)
            y = torch.reshape(y, [config.batch_size, 1])
            interval_popularity = interval_popularity.to(device)

            pred,nodes_pred = model(L,x,time,interval_popularity)
            loss = torch.mean(torch.pow((pred - y), 2))
            error = torch.mean(torch.pow((pred - y), 2))
            MAPE_error = torch.mean(torch.abs(pred-y)/(torch.log2(torch.pow(2,y)+1)+torch.log2(torch.pow(2,pred)+1))*2)
            torch.mean(torch.abs(pred-y)/torch.log2(torch.pow(2,y)+1))
            test_loss.append(error.item())
            test_MAPE.append(MAPE_error.item())

    print("-------------Final Test--------------")
    print("Test_MSLE:{:.3f},Test_MAPE:{:.3f}".format(np.mean(test_loss),np.mean(test_MAPE)))


if __name__ == '__main__':
    args = CasSampling_model.get_params()
    main(args)



