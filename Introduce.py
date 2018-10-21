import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import urllib.request

x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker="x")
plt.show()

person = {'Location':["Beijing","Hunan","Hebei","Los"],'Name':['Jack','Leo','Lee','Hu'],'Age':[24,22,44,21]}
data_pandas = pd.DataFrame(person)      # 数据结构:类似Excel的一张表。
display(data_pandas)

# 选择年龄大于22的所有行
display(data_pandas[data_pandas.Age > 22])