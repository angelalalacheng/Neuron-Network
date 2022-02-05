import numpy as np
import matplotlib.pyplot as plt


def draw(x_test, y_test, W):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    X1 = []
    Y1 = []
    X0 = []
    Y0 = []

    # 畫分散圖
    for i in range(0, len(x_test)):
        Arr = x_test[i]
        if y_test[i] == 1:
            X1.append(Arr[0][1])
            Y1.append(Arr[0][2])
        else:
            X0.append(Arr[0][1])
            Y0.append(Arr[0][2])

    X_Max_val = max(max(X1), max(X0))
    X_Min_val = min(min(X1), min(X0))

    ax.scatter(X1, Y1, marker="x")
    ax.scatter(X0, Y0, marker="o")

    # 畫出函式
    l = np.linspace(X_Min_val, X_Max_val, 100, dtype=float)

    if W[0][2] == 0.:
        a = W[0][0] / W[0][1]
        ax.plot(a*l, l, 'b-')

    else:
        b = W[0][0]/W[0][2]
        a = -W[0][1]/W[0][2]
        ax.plot(l, a*l+b, 'b-')
    plt.show()


def read_data(address):
    Input = open(address)
    data = Input.readlines()

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    # 分類train和test資料
    index = 0
    for line in data:
        buff = line.split(' ')
        if index % 3 == 0:
            try:
                # 才能transpose
                x_test.append(np.array([[-1, int(buff[0]), int(buff[1])]]))
            except:
                # 才能transpose
                x_test.append(np.array([[-1, float(buff[0]), float(buff[1])]]))

            y_test.append(int(buff[2]))

        else:
            try:
                x_train.append(np.array([[-1, int(buff[0]), int(buff[1])]]))
            except:
                x_train.append(
                    np.array([[-1, float(buff[0]), float(buff[1])]]))

            y_train.append(int(buff[2]))

        index += 1

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def func(x):
    if x >= 0:
        return 1
    else:
        return -1


def modify(w, x, Y, Yp):
    if(Y == 1 and Yp == -1):
        new_w = w-x*learning_rate
    elif(Y == -1 and Yp == 1):
        new_w = w+x*learning_rate
    else:
        new_w = w+x*learning_rate

    return new_w


def norm(List_d1, List_d2):
    Max = List_d1.max()
    Min = List_d1.min()
    Norm_d1 = (List_d1-Min)/(Max-Min)
    Norm_d2 = (List_d2-Min)/(Max-Min)

    for i in range(0, len(Norm_d1)):
        if Norm_d1[i] == 0:
            Norm_d1[i] = -1
    for i in range(0, len(Norm_d2)):
        if Norm_d2[i] == 0:
            Norm_d2[i] = -1

    return Norm_d1, Norm_d2


# 預設值
W = np.array([0, 0, 0])
learning_rate = 0.01
epoch = 100

# 輸入資料
address = input("輸入資料的位置:")
x_train, y_train, x_test, y_test = read_data(address)
Norm_y_train, Norm_y_test = norm(y_train, y_test)

learning_rate = float(input("輸入learning rate:"))
epoch = int(input("輸入epoch:"))

# training
for k in range(0, epoch):
    wrong = 0
    for i in range(0, len(x_train)):
        Trans_X = x_train[i].T
        V = W.dot(Trans_X)
        Y = func(V)

        if Y != Norm_y_train[i]:
            wrong += 1
            W = modify(W, x_train[i], Y, Norm_y_train[i])

print("training correct:", ((len(x_train)-wrong)/len(x_train))*100, "%")
print("W=", W)

# testing
correct = 0
for i in range(0, len(x_test)):
    Trans_X = x_test[i].T
    V = W.dot(Trans_X)
    Y = func(V)

    if Y == Norm_y_test[i]:
        correct += 1

print("testing correct:", (correct/len(x_test))*100, "%")

draw(x_test, Norm_y_test, W)
