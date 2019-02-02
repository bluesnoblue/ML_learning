import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data_sets/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values
# print('X:')
# print(X)
# print('Y:')
# print(Y)

# 拆分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)
# print('X_train:')
# print(X_train)
# print('X_test:')
# print(X_test)
# print('Y_train:')
# print(Y_train)
# print('Y_test:')
# print(Y_test)

#模型训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#预测结果
Y_pred = regressor.predict(X_test)

#训练集结果可视化
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.show()

#测试集结果可视化
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()
