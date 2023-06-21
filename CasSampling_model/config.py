import math

FLAG = "weibo"
DATA_PATHA = "../data/"+FLAG
if(FLAG=="weibo"):
    train_data = DATA_PATHA + "/data.pkl"
if(FLAG=="twitter"):
    train_data = DATA_PATHA + "/data.pkl"

# train_data = "E:/PyCharm Community Edition 2021.2.3/project/CasNode/data/data_train.pkl"

information = DATA_PATHA+"/information.pkl"

# train_data_aps = 'E:/PyCharm Community Edition 2021.2.3/project/CasNode/aps_data/data_train.pkl'

#parameters
# observation = 0.5*60*60-1
n_time_interval = 6
# print ("the number of time interval:",n_time_interval)
# time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整
# print ("time interval:",time_interval)


##model parameters
n_steps = 100
num_rnn_layers = 2
cl_decay_steps = 1000
num_kernel = 2
learning_rate = 0.005
batch_size = 64
num_hidden = 32
n_hidden_dense1 = 32
n_hidden_dense2 = 16
feat_in = 100
feat_out = 50
num_nodes = 100
drop_out_prob = 0.8
seed = 0


#GCN_model parameters


