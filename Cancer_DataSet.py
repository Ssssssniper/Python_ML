from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

"""
对数据集的获取需要熟练掌握下列方法
keys（）
shape
target_names
target
np.bincount(bunch.target)
"""

cancer = load_breast_cancer()   # bunch对象，可以使用.key直接访问对象，拥有类似字典的结构。
print("cancer.key():{}".format(cancer.keys()))
print("Shape of Dataset:{}".format(cancer.data.shape))
print("{}".format(cancer.target_names))     # y的名字
print("{}".format(cancer.target))           # 对应y名字的数据集
print("{}".format(np.bincount(cancer.target)))      # y的频次
print("{}".format(cancer.feature_names))        # x的特征名
print("Sample counts per class:{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("{}".format(cancer.DESCR))