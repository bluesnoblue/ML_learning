from numpy import inf
import random

def dist_elud(a,b):
    if a>b:
        return a-b
    else:
        return b-a

def rand_cent(data_set,k):
    center = [0]*k
    for i in range(k):
        range_j = float(max(data_set)-min(data_set))
        # print(range_j)
        center[i] = min(data_set) + range_j*random.random()
        # print(center)
    return  center

def k_means(data_set,k,dist=dist_elud,create_cent = rand_cent):
    m = len(data_set)
    cluster_assment = [[0,0] for z in range(m)]
    center = create_cent(data_set,k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):
                dist_ji = dist(data_set[i],center[j])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i][0] != min_index:
                cluster_changed = True
            cluster_assment[i][0] = min_index
            cluster_assment[i][1] = min_dist


        for cent_index in range(k):
            total = 0
            total_data = 0
            for i in range(m):
                if cluster_assment[i][0] == cent_index:
                    total_data += data_set[i]
                    total += 1
            center[cent_index] = total_data/total
        return cluster_assment

if __name__ == '__main__':
    # data_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 53, 55, 66, 67, 68, 69, 85, 80, 90, 100]
    data_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 53, 55, 66, 60, 60, 60, 85, 80, 90, 100]
    k = 3
    cluster_assment = k_means(data_set, k)
    for i in range(k):
        cluster = []
        for data_index in range(len(data_set)):
            if cluster_assment[data_index][0] == i:
                cluster.append(data_set[data_index])
        print(cluster)