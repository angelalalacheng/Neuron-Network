import numpy as np
import math
import matplotlib.pyplot as plt

# 神經元


class neuron:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, feature):
        v = self.weight.dot(feature)
        v = float(v)
        y = sigmoid(v)
        return y


# 網路

class Network:
    def __init__(self, hidden_layer_neuron, output_layer_neuron):
        self.hidden_layer = hidden_layer_neuron
        self.output = output_layer_neuron

    def feedforward(self, feature):
        hidden_layer_output = []
        for i in range(0, len(self.hidden_layer)):
            res = self.hidden_layer[i].forward(feature)
            hidden_layer_output.append(res)

        hidden_layer_output.insert(0, -1.)
        hidden_layer_output = np.array(hidden_layer_output)
        final = self.output.forward(hidden_layer_output)

        return hidden_layer_output, final

    def delta_hiddenlayer(self, delta_out, hidden_output):
        delta_hidden = []
        for i in range(1, len(hidden_output)):
            delta_hidden.append(
                (hidden_output[i])*(1-hidden_output[i])*delta_out*(self.output.weight[0][i]))
        return delta_hidden

    def delta_outputlayer(self, d, predict):
        delta_out = (d-predict)*predict*(1-predict)
        return delta_out

    def modify_out_w(self, learning_rate, delta, hidden_output):
        mod_out_w = learning_rate*delta*hidden_output
        mod_out_w = mod_out_w.reshape(1, 9)
        return mod_out_w

    def modify_hidden_w(self, learning_rate, delta, feature):
        tmp = feature.reshape(1, 4)
        mod_hidden_w = []
        for i in range(0, len(delta)):
            mod_hidden_w.append(learning_rate*delta[i]*tmp)
        return mod_hidden_w

    def train(self, true, features):
        learning_rate = 0.5
        epoch = 100
        for i in range(epoch):
            for j in range(0, len(true)):
                # forward
                hidden_output, predict = self.feedforward(features[j])

                if predict != true[j]:
                    # update output layer
                    delta_out = self.delta_outputlayer(true[j], predict)

                    mod_out_w = self.modify_out_w(
                        learning_rate, delta_out, hidden_output)

                    # update hidden layer
                    delta_hidden = self.delta_hiddenlayer(
                        delta_out, hidden_output)

                    mod_hidden_w = self.modify_hidden_w(
                        learning_rate, delta_hidden, features[j])

                    for k in range(0, len(self.hidden_layer)):
                        self.hidden_layer[k].weight = self.hidden_layer[k].weight + \
                            mod_hidden_w[k]

                    self.output.weight = self.output.weight+mod_out_w
                    # calculate loss

        return self.hidden_layer, self.output.weight


def test(hiddenlayer_weight, outputlayer_weight, parms, test_data):
    test_hidden_out = []
    test_data.insert(0, -1)
    test_arr = np.array(test_data)
    test_arr = test_arr.astype(float)

    for i in range(1, 4):
        test_arr[i] = (test_arr[i]-parms[1])/(parms[0]-parms[1])

    for j in range(0, len(hiddenlayer_weight)):
        h_out = sigmoid(hiddenlayer_weight[j].weight.dot(test_arr))
        test_hidden_out.append(h_out)

    test_hidden_out.insert(0, -1.)
    test_hidden_out_arr = np.array(test_hidden_out)

    angle = sigmoid(outputlayer_weight.dot(test_hidden_out_arr))
    angle_recover = angle*(parms[2]-parms[3])+parms[3]
    # f.write(str(angle_recover)+'\n')
    return angle_recover


def sigmoid(v):
    res = 1/(1+math.exp((-1)*v))
    return res


def draw(LOSS):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(LOSS)
    plt.show()


def read_data(address):
    Input = open(address)
    data = Input.readlines()
    x_train = []
    x_test = []
    a = []
    b = []
    for line in data:
        buff = line.split(' ')
        for i in range(0, 4):
            if i == 3:
                b.append(float(buff[i]))
            else:
                a.append(float(buff[i]))

    a = np.array(a)
    b = np.array(b)

    train_max = a.max()
    test_max = b.max()
    train_min = a.min()
    test_min = b.min()

    for line in data:
        buff = line.split(' ')

        stringArr = np.array([-1., buff[0], buff[1], buff[2]])
        floatArr = stringArr.astype(float)
        for i in range(1, 4):
            floatArr[i] = (floatArr[i]-train_min)/(train_max-train_min)
        x_train.append(floatArr)

        stringD = buff[3]
        floatD = float(stringD)
        x_test.append((floatD-test_min)/(test_max-test_min))

    return x_train, x_test, train_max, train_min, test_max, test_min


# main
parms = []
hidden_layer_neuron = []
for i in range(0, 8):
    hidden_layer_neuron.append(neuron(np.random.rand(1, 4)))

output_layer_neuron = neuron(np.random.rand(1, 9))

network = Network(hidden_layer_neuron, output_layer_neuron)

features, d, train_max, train_min, test_max, test_min = read_data(
    'C:\\Users\\angela_cheng\\Downloads\\NN+HW2_Dataset\\train4dAll.txt')

parms.append(train_max)
parms.append(train_min)
parms.append(test_max)
parms.append(test_min)

hiddenlayer_weight, outputlayer_weight = network.train(d, features)
'''
path = 'output.txt'
f = open(path, 'w')
for i in range(0, len(features)):
    test(hiddenlayer_weight, outputlayer_weight,
         parms, test_data=features[i])
f.close()
'''
