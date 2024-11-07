import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 선형 회귀 모델

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
yt = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

print(perch_length.shape)
print(yt.shape)

perch_length=perch_length.reshape(-1,1)

print(perch_length)
x=np.insert(perch_length, 0, 1.0, axis=1)
print(x)

plt.scatter(x[:,1], yt)
print(x.shape)
plt.plot([15,50],[15*16.+1, 50*16.8+1])
plt.plot([15,50],[15*36-648, 50*36-648])
plt.show()

def pred(x, w):
    return(x @ w)

# 초기화 처리

# 데이터 전체 건수
M = x.shape[0]

# 입력 데이터의 차수 (더미 변수 포함)
D = x.shape[1]

# 반복 횟수
iters = 3000000

# 학습률
alpha = 0.001

# 가중치 벡터의 초깃값 (모든 값을 1로 한다)
w = np.ones(D)

# 평가 결과 기록 (손실함수의 값만 기록)
history = np.zeros((0,2))

# 반복 루프
for k in range(iters):

    # 예측값 계산 (7.8.1)
    yp = pred(x, w)

    # 오차 계산 (7.8.2)
    yd = yp - yt

    # 경사하강법 적용 (7.8.4)
    w = w - alpha * (x.T.dot( yd)) / M

    # 학습 곡선을 그리기 위한 데이터 계산 및 저장
    if (k % 1000 == 0):
        # 손실함숫값의 계산 (7.6.1)
        loss = np.mean(yd ** 2) / 2
        # 계산 결과의 기록
        history = np.vstack((history, np.array([k, loss])))
        # 화면 표시
        print("iter=%d loss=%f" %(k, loss))

print(yd, loss)
print(yd.shape)
print(w)

lr=LinearRegression()
lr.fit(perch_length, yt)
print(lr.coef_, lr.intercept_)