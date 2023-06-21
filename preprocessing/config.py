import math
DATA_PATHA = "../data"
FLAG = "weibo"
DATA_PATHA = "../data/"+FLAG
max_NumNode = 128
n_time_interval = 64

if(FLAG == "weibo"):
    cascades = DATA_PATHA + "/dataset_weibo.txt"
    cascade_train = DATA_PATHA + "/cascade.txt"
    shortestpath_train = DATA_PATHA + "/shortestpath.txt"
    longpath = DATA_PATHA + "/longpath.txt"

    observation = 0.5 * 60 * 60 - 1
    time_interval = math.ceil((observation + 1) * 1.0 / n_time_interval)  # 向上取整

    pre_times = [24 * 3600]
    train_pkl = DATA_PATHA + "/data.pkl"

if(FLAG=="twitter"):

    cascades = DATA_PATHA + "/dataset.txt"
    cascade_train = DATA_PATHA + "/cascade.txt"
    shortestpath_train = DATA_PATHA + "/shortestpath.txt"

    observation = 3600*24*2
    time_interval = math.ceil((observation + 1) * 1.0 / n_time_interval)
    pre_times = [2764800]
    train_pkl = DATA_PATHA + "/data.pkl"

print("dataset:",FLAG)
print("observation time", observation)
print("the number of time slots:", n_time_interval)



