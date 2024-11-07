import numpy as np
import matplotlib.pyplot as plt

eta = 0.01 # 학습률
x = 3 # 시작점
record_x = []
record_y = []

for i in range(2000):
    y = 3*x**2-6*x+11/3 # 최솟값을 구할 식
    record_x.append(x)
    record_y.append(y)
    x = x-eta*(6*x-6) # x를 y를 미분한 값에 학습률을 곱해 이동
xx = np.linspace(-2, 4)
yy = 3*xx**2-6*xx+11/3

plt.plot(xx, yy, c = 'r', linestyle = 'dashed')
plt.scatter(record_x, record_y)
plt.show()

print(record_y[len(record_y) - 1])