import pandas as pd

data_set = pd.read_csv('data_sets/50_Start_ups.csv')
X = data_set.iloc[ : , :-1].values
Y = data_set.iloc[ : ,  4 ].values

#将类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[: , 3] = label_encoder.fit_transform(X[ : , 3])
one_hot_encoder = OneHotEncoder(categorical_features = [3])
X = one_hot_encoder.fit_transform(X).toarray()

#躲避虚拟变量陷阱 去掉了第二列
X = X[: , 1:]


# 拆分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#模型训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 在测试集上预测结果
y_pred = regressor.predict(X_test)
print('Y_test:')
print(Y_test)
print('y_pred:')
print(y_pred)

from sklearn.metrics import r2_score #R2 决定系数（拟合优度）
print(r2_score(Y_test,y_pred))


