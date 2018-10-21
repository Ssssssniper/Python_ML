from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 包含键值对
iris_dataset = load_iris()

# 训练集和测试集的拆分----------train_test_split()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)
print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))

print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_name中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵,按y_train着色
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                                hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

# KNN算法进行分类
knn = KNeighborsClassifier(n_neighbors=1)       # 实例化，并设定邻居数目
knn.fit(X_train, y_train)                       # 构建模型
print(knn.fit(X_train, y_train))

# 用此模型对新数据进行预测
X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape{}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))

# 评估模型(利用测试集来进行评估,算精度给模型评价)
y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))
print(y_test)
print("Test set score:{:.2f}".format(np.mean(y_pred == y_test)))      # 计算预测准确的占比
print("Test set score:{:.2f}".format(knn.score(X_test, y_test)))      # knn.score()计算测试集的精度
