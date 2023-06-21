import time
from CasSampling.preprocessing import config
import re


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


def gen_cascades_obser(observation_time,pre_times,filename):
    cascades_total = dict()
    cascades_type = dict()
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            if(FLAG == "weibo"):
                path = parts[4].split(" ")
                if len(parts) != 5:
                    print('wrong format!')
                    continue
            if (FLAG == "twitter"):
                path = parts[4].split(" ")
                path = path[1:]
                if len(parts) != 5:
                    print('wrong format!')
                    continue
            cascadeID = parts[0]
            n_nodes = int(parts[3])
            if n_nodes != len(path):
                print('wrong number of nodes', n_nodes, len(path))
            msg_pub_time = parts[2]

            observation_path = []
            labels = []
            edges = set()
            for i in range(len(pre_times)):
                labels.append(0)
            for p in path:
                nodes = p.split(":")[0].split("/")
                nodes_ok = True
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                for i in range(len(pre_times)):
                    if time_now < pre_times[i]:
                        labels[i] += 1
            cascades_total[cascadeID] = msg_pub_time

        n_total = len(cascades_total)
        print('total:', n_total)


        count = 0
        for k in cascades_total:
            if count < n_total * 1.0 / 20 * 14:
                cascades_type[k] = 1
            elif count < n_total * 1.0 / 20 *17:
                cascades_type[k] = 2
            else:
                cascades_type[k] = 3
            count += 1
    return cascades_total,cascades_type


def discard_cascade(observation_time,pre_times,filename):
    discard_cascade_id=dict()
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            if (FLAG == "weibo"):
                path = parts[4].split(" ")
                if len(parts) != 5:
                    print('wrong format!')
                    continue
            if (FLAG == "twitter"):
                path = parts[4].split(" ")
                path = path[1:]
                if len(parts) != 5:
                    print('wrong format!')
                    continue

            cascadeID = parts[0]
            n_nodes = int(parts[3])
            # path = parts[4].split(" ")
            if n_nodes != len(path):
                print('wrong number of nodes', n_nodes, len(path))
            msg_pub_time = parts[2]

            observation_path = []
            edges = set()
            for p in path:
                nodes = p.split(":")[0].split("/")
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                        
            if len(observation_path)<10:
                discard_cascade_id[cascadeID] = 1
                continue;

            else:
                discard_cascade_id[cascadeID]=0


    return discard_cascade_id



def gen_cascade(observation_time, pre_times, filename, filename_ctrain, filename_strain,cascades_type, discard_cascade_id):
    file = open(filename,"r")
    file_ctrain = open(filename_ctrain, "w")
    file_strain = open(filename_strain, "w")

    for line in file:
        parts = line.split("\t")
        if (FLAG == "weibo"):
            path = parts[4].split(" ")
            if len(parts) != 5:
                print('wrong format!')
                continue

        if (FLAG == "twitter"):
            path = parts[4].split(" ")
            path = path[1:]
            if len(parts) != 5:
                print('wrong format!')
                continue


        cascadeID = parts[0]

        n_nodes = int(parts[3])
        # path = parts[4].split(" ")
        if n_nodes !=len(path):
            print ('wrong number of nodes',n_nodes,len(path))
        # msg_time = time.localtime(int(parts[2]))
        if(FLAG =="weibo"):
            msg_time = time.localtime(int(parts[2]))
            hour = time.strftime("%H",msg_time)
            hour = int(hour)
            if hour <= 7 or hour >= 18:
                continue
        if(FLAG == "twitter"):
            month = int(time.strftime('%m', time.localtime(float(parts[2]))))
            day = int(time.strftime('%d', time.localtime(float(parts[2]))))
            if month == 4 and day > 10:
                continue
        observation_path = []
        all_path = []
        labels = []
        edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            time_now = int(p.split(":")[1])
            all_path.append(",".join(nodes) + ":" + str(time_now))
            if time_now <observation_time:
                observation_path.append(",".join(nodes)+":"+ str(time_now))
                for i in range(1,len(nodes)):
                    if (nodes[i-1] +":"+ nodes[i] +":"+ str(time_now)) in edges:
                        continue
                    else:
                        edges.add(nodes[i-1]+":"+ nodes[i]+":"+ str(time_now))
            for i in range(len(pre_times)):
                if time_now <pre_times[i]:
                    labels[i] +=1
        for i in range(len(labels)):
            labels[i] = str(labels[i]-len(observation_path))
        if len(edges)<=1:
            continue
        if cascadeID in cascades_type and discard_cascade_id[cascadeID]== 0:
                file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")    #shortespath_train
                file_ctrain.write(cascadeID+"\t"+parts[1]+"\t"+parts[2]+"\t"+str(len(observation_path))+"\t"+" ".join(edges)+"\t"+" ".join(labels)+"\n")
    file.close()
    file_ctrain.close()
    file_strain.close()

if __name__ =="__main__":

    FLAG = config.FLAG
    observation_time = config.observation
    pre_times = config.pre_times

    cascades_total, cascades_type = gen_cascades_obser(observation_time,pre_times,config.cascades)
    discard_cascade_id= discard_cascade(observation_time,pre_times,config.cascades)

    print("generate cascade new!!!")
    gen_cascade(observation_time, pre_times, config.cascades, config.cascade_train,config.shortestpath_train,cascades_type, discard_cascade_id)







