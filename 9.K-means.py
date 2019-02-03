import pandas as pd
from numpy import *



# 计算欧氏距离
def dist_elud(vec_a,vec_b):
    return sqrt(sum(power((vec_a - vec_b),2)))

# 初始化聚类中心
def rand_cent(data_set,k):
    n = shape(data_set)[1] #因素个数
    center = mat(zeros((k,n)))
    for j  in range(n): #遍历因素
        range_j = float(max(data_set[:,j])-min(data_set[:,j])) # 因素取值范围
        center[:,j] = min(data_set[:,j]) + range_j * random.rand(k,1)
    return center

def k_means(data_set, k, dist = dist_elud,create_cent = rand_cent):
    m = shape(data_set)[0]  # 数据个数
    cluster_assment = mat(zeros((m,2))) # 生成一个与data_set大小一致的矩阵
    center = create_cent(data_set,k) # 获取k个随机中心
    cluster_changed = True #cluster是否改变
    while cluster_changed:
        cluster_changed = False
        for i in range(m): #遍历每个数据
            min_dist = inf  #最小欧氏距离 暂取无穷大
            min_index = -1  #欧氏距离最小的中心点的角标
            for j in range(k):    #遍历每个中心
                dist_ji = dist(data_set[i,:],center[j,:]) #计算每个 数据 与 中心 的欧氏距离
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j   #欧氏距离最小的中心点的角标

            if cluster_assment[i,0] != min_index: #判断是否收敛（判断每个点是否更新）
                cluster_changed = True

            cluster_assment[i,:] = min_index,min_dist ** 2

        # print(center)
        #移动中心点 下面的还没看懂
        for cent in range(k):
            data_cent = data_set[nonzero(cluster_assment[:,0].A ==cent)[0]]
            center[cent,:] = mean(data_cent,axis=0)

    return center,cluster_assment

if __name__ == '__main__':
    data_set = pd.read_csv('data_sets/Social_Network_Ads.csv')
    data_set = data_set.iloc[:, [2, 3]].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    data_set = sc.fit_transform(data_set)
    # print(rand_cent(data_set,2))
    a = k_means(data_set,2)
    print('center')
    print(a[0])
    print('cluster_assment')
    print(a[1])
