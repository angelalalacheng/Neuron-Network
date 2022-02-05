import numpy as np


# 讀檔


def read_data(address):
    Read = open(address)
    data = Read.readlines()
    DATAs = []
    input = []
    count = 0
    P = 0
    for line in data:
        if line == '\n':
            continue

        for i in range(0, len(line)):
            if line[i] == ' ':
                input.append(-1.)
            if line[i] == '1':
                input.append(1.)
        count += 1

        if count == 12:
            P = len(input)
            input_arr = np.array([input])
            DATAs.append(input_arr)
            count = 0
            input.clear()

    return P, DATAs

# 計算 W


def cal_W():
    # default:'C:\\Users\\angela_cheng\\Downloads\\Hopfield_dataset\\Basic_Training.txt'
    Add1 = input('training data address:')
    P, training_data = read_data(Add1)
    N = len(training_data)
    res = np.zeros((P, P), dtype=float)
    I = np.identity(P)

    for i in range(0, N):
        tmp = np.transpose(training_data[i]).dot(training_data[i])
        res = res + tmp

    W = (res*1-I*N)/P

    return W

# 計算閥值


def cal_theta():
    W = cal_W()
    theta = []
    for i in range(0, W.shape[0]):
        element = 0
        for j in range(0, W.shape[0]):
            element += W[i][j]

        theta.append(element)

    return W, theta

# sign function


def sign(num, orign):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return orign

# 訓練


def train():
    W, theta = cal_theta()
    return W, theta

# 畫結果


def draw(arr):
    arr = arr.flatten('C')
    L = arr.tolist()
    for i in range(0, len(L)):
        if L[i] == 1.0:
            print('▮', end='')
        if L[i] == -1.0:
            print('▯', end='')
        if i % 9 == 8:
            print('\n', end='')
# 測試


def test():
    # default:'C:\\Users\\angela_cheng\\Downloads\\Hopfield_dataset\\Basic_Testing.txt'
    Add2 = input('testing data address:')
    P, testing_data = read_data(Add2)
    W, get_theta = train()
    theta = np.array([get_theta])

    before = 0
    after = 0
    for test_index in range(0, len(testing_data)):
        stop = False
        while stop == False:
            cal = 0
            before = testing_data[test_index]
            for i in range(0, testing_data[0].shape[1]):
                re_row = W[i].reshape(-1, 108)
                tmp = np.multiply(re_row, testing_data[test_index])
                orign = testing_data[test_index][0][i]
                for j in range(0, testing_data[0].shape[1]):
                    cal += tmp[0][j]
                res = sign(cal-theta[0][i], orign)
                testing_data[test_index][0][i] = res
                cal = 0
            after = testing_data[test_index]
            stop = (before == after).all()
        print('testcase', test_index+1)
        draw(after)


# main
test()
