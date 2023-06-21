import numpy as np
import six.moves.cPickle as pickle
from CasSampling.preprocessing import config
import networkx as nx
import scipy.sparse
import gc
import math
LABEL_NUM = 0
from tqdm.auto import tqdm

class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)

#transform the sequence to list
def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = [] #walk[0] = cascadeID
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0] #node
                t = walks[i].split(":")[1] #time
                graphs[walks[0]].append([[str(xx) for xx in s.split(",")],int(t)])
    return graphs

#read label and size from cascade file
def read_labelANDsize(filename):
    labels = {}
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            labels[profile[0]] = profile[-1]
            sizes[profile[0]] = int(profile[3])
    return labels,sizes

def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print ("length of original isd:",len(original_ids))
    return original_ids

def get_nodes(graph):
    nodes = {}
    j = 0
    for walk in graph:
        for i in walk[0]:
            if i not in nodes.keys():
                nodes[i] = j
                j = j+1
    return nodes

def write_data(graphs,labels,sizes,index,max_num, filename):
    #get the data for model
    id_data = []
    x_data = []
    y_data = []
    sz_data = []
    time_data = []
    adjacency_data = []
    interval_popularity_data=[]
    avg_sequence = 0
    for graph in tqdm(graphs.items()):
        key,graph = graph
        id = key
        label = labels[key].split()
        y = int(label[LABEL_NUM]) #label
        temp = []
        temp_time = [] #store time
        size_temp = len(graph)
        graph = sorted(graph, key=lambda x: x[1])
        if size_temp !=  sizes[key]:
            print (size_temp,sizes[key])
        nodes_items = get_nodes(graph)
        nodes_list = nodes_items.values()         ###获得节点列表
        number2node = {}
        for key,value in nodes_items.items():
            number2node[value] = key
        nx_G = nx.DiGraph()

        Len = 0
        degree = np.array(np.zeros(size_temp))
        i = 0

        sequence = 0
        for walk in graph:
             sequence = sequence+(len(walk[0]))
        sequence = sequence/len(graph)
        avg_sequence += sequence

        for walk in graph:
            temp.append(walk[1])
            if i == 0:
                node = str(walk[0][len(walk[0]) - 1])
                i = i + 1
            else:
                node = str(walk[0][len(walk[0]) - 2])
            degree[nodes_items[node]] = degree[nodes_items[node]] + 1   ###compute node degree

        time_interval_popularity = np.array(np.zeros([config.n_time_interval]))

        for i in temp:
            interval  = int(math.floor(i/config.time_interval))
            time_interval_popularity[interval] = time_interval_popularity[interval] + 1

        time_dict = dict(zip(nodes_list,temp))
        degree_dict = dict(zip(nodes_list,degree))
        degree_dict =sorted(degree_dict.items(),key=lambda x:(x[1],x[0]),reverse=True)

        degree_dict = dict(degree_dict)


        if(size_temp<max_num):
            node_list = list(degree_dict.keys())
            node_feature = list(degree_dict.values())
        else:
            node_list = list(degree_dict.keys())[0:max_num]
            node_feature = list(degree_dict.values())[0:max_num]
        nx_G.add_nodes_from(node_list)

        for i in node_list:
            temp_time.append(time_dict[i])

        i=0
        while i<len(graph):
            if (nodes_items.get(graph[i][0][-1]) in node_list):
                i=i+1
            else:
                graph.pop(i)

        for walk in graph:
            ####create adj###
            Len = Len + 1
            walk_time = walk[1]
            if walk_time == 0:
                nx_G.add_edge(nodes_items.get(walk[0][0]), nodes_items.get(walk[0][0]))

            for i in range(len(walk[0])-1):
                if(nodes_items.get(walk[0][len(walk[0])-2-i]) in node_list):
                    nx_G.add_edge(nodes_items.get(walk[0][len(walk[0])-2-i]),nodes_items.get(walk[0][-1]))
                    nx_G.add_edge(nodes_items.get(walk[0][-1]),nodes_items.get(walk[0][-1]))
                    break
            if (Len != len(graph)):
                continue

            temp_adj = nx.to_pandas_adjacency(nx_G)
            degree_adj = np.array(temp_adj)
            ####create adj###

            node_feature = np.array(node_feature)
            node_feature = np.log2(node_feature + 2) ###add 2 to distinguish it from padding
            N = len(temp_adj)
            for i in range(N):
                degree_adj[i][i] = node_feature[i]
            node_feature = degree_adj


            temp_time = np.array(temp_time)/config.observation
            time_interval_popularity = np.log2(time_interval_popularity+1)/10
            ###padding
            if N <= max_num:
                padding = np.zeros(shape=(max_num-N))
                temp_time = np.concatenate((temp_time,padding))
                col_padding = np.zeros(shape=(N, max_num - N))
                A_col_padding = np.column_stack((temp_adj, col_padding))
                row_padding = np.zeros(shape=(max_num - N, max_num))
                node_feature = np.column_stack((node_feature, col_padding))
                node_feature = np.concatenate((node_feature, row_padding))
                node_feature = scipy.sparse.coo_matrix(node_feature, dtype=np.float32)
                A_col_row_padding = np.row_stack((A_col_padding, row_padding))
                temp_adj = scipy.sparse.coo_matrix(A_col_row_padding, dtype=np.float32)
            else:
                temp_adj = scipy.sparse.coo_matrix(temp_adj,dtype=np.float32)
                node_feature = scipy.sparse.coo_matrix(node_feature, dtype=np.float32)



        time_data.append(temp_time)
        id_data.append(id)
        x_data.append(node_feature)
        y_data.append(np.log2(y+1.0))
        adjacency_data.append(temp_adj)
        interval_popularity_data.append(time_interval_popularity)
        sz_data.append(size_temp)
    gc.collect()
    avg_sequence = avg_sequence/len(graphs)
    # with open(filename, 'wb') as f:
    pickle.dump((id_data,x_data,adjacency_data,y_data, sz_data, time_data,interval_popularity_data,index.length()), open(filename,'wb'))



if __name__ == "__main__":
    FLAG = config.FLAG

    shortestpath_train = config.shortestpath_train
    cascade_train = config.cascade_train
    train_pkl = config.train_pkl

    ### data set ###
    graphs_train = sequence2list(shortestpath_train)

    labels_train, sizes_train = read_labelANDsize(cascade_train)


    max_num = config.max_NumNode
    original_ids = get_original_ids(graphs_train)

    original_ids.add(-1)
    ## index is new index
    index = IndexDict(original_ids)

    print("create train")
    write_data(graphs_train,labels_train,sizes_train,index,max_num, train_pkl)
    #pickle.dump((len(original_ids),NUM_SEQUENCE,LEN_SEQUENCE), open(config.information,'wb'))
    print("Finish!!!")

