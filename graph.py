from matplotlib import pyplot as plt
from random import randint
import scipy



data = scipy.io.loadmat('data/matlab.mat')
a = data['ds'] #matに保存した変数の呼び出し

print(a)

# データの定義(サンプルなのでテキトー)
x = a
y1 = [randint(0, 100) for _ in x]
y2 = [randint(0, 100) for _ in x]

# グラフの描画
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
""""
#text = open('data/text.txt')
path = 'data/text.txt'

f = open(path)

print(type(f))
print(f)
# <class '_io.TextIOWrapper'>

f.close()

"""
"""
# データの定義(サンプルなのでテキトー)
x = list(range(10))
y1 = [randint(0, 100) for _ in x]
y2 = [randint(0, 100) for _ in x]

# グラフの描画
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
"""