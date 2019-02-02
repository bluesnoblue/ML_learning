import pandas as pd

# 导入数据
data_set = pd.read_csv('data_sets/Data.csv')
X = data_set.iloc[:,:-1].values #要查一下.iloc[:,:-1]是什么
Y = data_set.iloc[:,3].values

# print('-----导入数据-----')
# print(data_set)
# print(X)
# print(Y)

# 处理丢失数据
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
# print('-----处理丢失数据-----') # 缺失的数据填入均值
# print('X:')
# print(X)

# 解析 分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # 向量化或者数字化
label_encoder_X = LabelEncoder() # 标签编码
X[:,0] = label_encoder_X.fit_transform(X[:,0])  # 对第一列  标签编码

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)
# print('-----解析 分类数据-----') # 把str名字转成int数字
# print('X:')
# print(X)
# print('Y:')
# print(Y)

# 创建虚拟变量
one_hot_encoder = OneHotEncoder(categorical_features = [0]) # 对第一列 独热编码
X = one_hot_encoder.fit_transform(X).toarray()

# print('-----创建虚拟变量-----')
# print('X:')
# print(X)


#拆分数据集未训练集合和测试集合
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state= 0)
# print('-----拆分数据集-----')
print('X_train:')
print(X_train)
print('X_test:')
print(X_test)
# print('Y_train:')
# print(Y_train)
# print('Y_test:')
# print(Y_test)

#特征缩放
from sklearn.preprocessing import StandardScaler  # 去均值和方差归一化
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# print('-----特征量化-----')
print('X_train:')
print(X_train)
print('X_test:')
print(X_test)
