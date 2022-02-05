import os
import sys
import math
import tkinter
import random
from NN_HW2 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def line_intersection(line1, line2):
    x1 = line1[0][0]
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]
    k1, k2, b1, b2 = None, None, None, None
    if(x2-x1 == 0):
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
    if(x4-x3 == 0):
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 == None and k2 == None:
        return
    elif k1 == None:
        x = x1
        y = k2 * x * 1.0 + b2 * 1.0
    elif k2 == None:
        x = x3
        y = k1 * x * 1.0 + b1 * 1.0
    elif k1 - k2 == 0:
        return
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    return round(x, 12), round(y, 12)


def get_point_on_body(position, angle, radius):
    return (position[0]+radius*mycos(angle), position[1]+radius*mysin(angle))


def mycos(angle):
    if(angle == 90 or angle == 270):
        return 0
    else:
        return math.cos(math.radians(angle))


def mysin(angle):
    if(angle == 0 or angle == 180):
        return 0
    else:
        return math.sin(math.radians(angle))


def cal_dis(position, wi):
    return round(math.dist(position, wi), 7)


class Car:
    def __init__(self, init_position, init_angle, verts, fourdatas):

        self.position = init_position
        self.angle = init_angle
        self.radius = 3
        self.alive = True
        self.arrival = False
        self.tracefour = ""
        self.tracesix = ""
        self.walls = []
        self.fourdatas = fourdatas
        for i in range(len(verts)-1):
            self.walls.append(verts[i]+verts[i+1])
        self.body_intersection = []
        self.wall_intersection = []

        self.body_intersection.append(get_point_on_body(
            self.position, self.angle, self.radius))
        self.body_intersection.append(get_point_on_body(
            self.position, self.angle-45, self.radius))
        self.body_intersection.append(get_point_on_body(
            self.position, self.angle+45, self.radius))
        self.wall_intersection.append(self.body_intersection[0])
        self.wall_intersection.append(self.body_intersection[1])
        self.wall_intersection.append(self.body_intersection[2])
        self.update_sensor()
        currentstate = [cal_dis(self.position, self.wall_intersection[0]), cal_dis(
            self.position, self.wall_intersection[1]), cal_dis(self.position, self.wall_intersection[2]), -1]
        self.tracefour = self.tracefour + str(round(currentstate[0], 7)) + " " + str(round(
            currentstate[1], 7)) + " " + str(round(currentstate[2], 7)) + " " + str(round(0, 7)) + "\n"
        self.tracesix = self.tracesix + str(round(self.position[0], 7)) + " " + str(round(self.position[1], 7)) + " " + str(round(
            currentstate[0], 7)) + " " + str(round(currentstate[1], 7)) + " " + str(round(currentstate[2], 7)) + " " + str(round(0, 7)) + "\n"

    def draw(self, ax):
        self.body, = ax.plot(self.position[0], self.position[1], color='blue', marker='o',
                             markersize=np.pi*self.radius**2, markeredgecolor='black', linestyle='')
        self.front_sensor, = ax.plot([self.body_intersection[0][0], self.wall_intersection[0][0]], [
                                     self.body_intersection[0][1], self.wall_intersection[0][1]], color='black')
        self.right_sensor, = ax.plot([self.body_intersection[1][0], self.wall_intersection[1][0]], [
                                     self.body_intersection[1][1], self.wall_intersection[1][1]], color='black')
        self.left_sensor, = ax.plot([self.body_intersection[2][0], self.wall_intersection[2][0]], [
                                    self.body_intersection[2][1], self.wall_intersection[2][1]], color='black')
        self.empty = ax.text(
            self.position[0], self.position[1], "", fontsize=6)
        self.sensor_dis = []
        for wi in self.wall_intersection:
            tmp = ax.text(wi[0], wi[1], str(
                cal_dis(self.position, wi)), fontsize=6)
            self.sensor_dis.append(tmp)
        self.pt = ax.text(self.position[0], self.position[1], "("+str(
            round(self.position[0]))+","+str(round(self.position[1]))+")", fontsize=6)

    def update(self, fig):
        self.ani = animation.FuncAnimation(
            fig=fig, func=self.update_ani, frames=500, interval=100, blit=True, repeat=False)

    def update_ani(self, i):
        if self.alive and not self.arrival:
            self.body.set_data(self.position[0], self.position[1])
            self.front_sensor.set_xdata(
                [self.body_intersection[0][0], self.wall_intersection[0][0]])
            self.front_sensor.set_ydata(
                [self.body_intersection[0][1], self.wall_intersection[0][1]])
            self.right_sensor.set_xdata(
                [self.body_intersection[1][0], self.wall_intersection[1][0]])
            self.right_sensor.set_ydata(
                [self.body_intersection[1][1], self.wall_intersection[1][1]])
            self.left_sensor.set_xdata(
                [self.body_intersection[2][0], self.wall_intersection[2][0]])
            self.left_sensor.set_ydata(
                [self.body_intersection[2][1], self.wall_intersection[2][1]])
            for index in range(len(self.wall_intersection)):
                self.sensor_dis[index].set_position(
                    self.wall_intersection[index])
                self.sensor_dis[index].set_text(
                    str(cal_dis(self.position, self.wall_intersection[index])))
            self.pt.set_position(self.position)
            self.pt.set_text(
                "("+str(round(self.position[0]))+","+str(round(self.position[1]))+")")

            currentstate = [cal_dis(self.position, self.wall_intersection[0]), cal_dis(
                self.position, self.wall_intersection[1]), cal_dis(self.position, self.wall_intersection[2])]

            self.update_position(currentstate)
            self.update_sensor()
            return (self.body, self.front_sensor, self.left_sensor, self.right_sensor, self.pt, self.sensor_dis[0], self.sensor_dis[1], self.sensor_dis[2])
        return (self.empty,)

    def update_position(self, currentstate):
        # swa MLP為預測角度
        # swa = random.uniform(-40, 40)
        #swa = int(self.mlp.predict(currentstate))
        swa = test(hiddenlayer_weight, outputlayer_weight,
                   parms, test_data=currentstate)
        print("predict:", swa)
        self.position[0] = self.position[0] + \
            mycos(self.angle + swa) + mysin(swa)*mysin(self.angle)
        self.position[1] = self.position[1] + \
            mysin(self.angle + swa) - mysin(swa)*mycos(self.angle)
        b = 2 * self.radius
        self.angle = self.angle - math.degrees(math.asin(2*mysin(swa)/b))
        self.body_intersection[0] = get_point_on_body(
            self.position, self.angle, self.radius)
        self.body_intersection[1] = get_point_on_body(
            self.position, self.angle-45, self.radius)
        self.body_intersection[2] = get_point_on_body(
            self.position, self.angle+45, self.radius)
        self.tracefour = self.tracefour + str(round(currentstate[1], 7)) + " " + str(round(
            currentstate[2], 7)) + " " + str(round(currentstate[3], 7)) + " " + str(round(swa, 7)) + "\n"
        self.tracesix = self.tracesix + str(round(self.position[0], 7)) + " " + str(round(self.position[1], 7)) + " " + str(round(
            currentstate[1], 7)) + " " + str(round(currentstate[2], 7)) + " " + str(round(currentstate[3], 7)) + " " + str(round(swa, 7)) + "\n"
        self.check_collision()
        self.check_arrival()

    def update_sensor(self):
        for i in range(len(self.wall_intersection)):
            front_line = (tuple(self.position), self.body_intersection[i])
            min_distance = sys.maxsize
            for wall in self.walls:
                wall_line = (([wall[0], wall[1]]), ([wall[2], wall[3]]))
                intersection_point = line_intersection(front_line, wall_line)
                Pi = wall_line[0]
                Pj = wall_line[1]
                if(intersection_point):
                    Q = list(intersection_point)
                    if((Q[0]-Pi[0])*(Pj[1]-Pi[1]) == (Pj[0]-Pi[0])*(Q[1]-Pi[1])
                       and min(Pi[0], Pj[0]) <= Q[0]
                       and min(Pi[1], Pj[1]) <= Q[1]
                       and Q[0] <= max(Pi[0], Pj[0])
                       and Q[1] <= max(Pi[1], Pj[1])
                       and np.dot((np.array(intersection_point)-np.array((self.position))), (np.array(self.body_intersection[i])-np.array(self.position))) > 0
                       and math.dist(intersection_point, self.position) <= min_distance
                       ):
                        min_distance = math.dist(
                            intersection_point, self.position)
                        self.wall_intersection[i] = intersection_point

    def check_arrival(self):
        self.arrival = False
        if(self.position[0] >= finish_line_UL[0] and self.position[0] <= finish_line_LR[0] and self.position[1] >= finish_line_LR[1] and self.position[1] <= finish_line_UL[1]):
            self.arrival = True
            f = open('train4D.txt', 'w')
            f.write(self.tracefour)
            f.close()
            f = open('train6D.txt', 'w')
            f.write(self.tracesix)
            f.close()

    def check_collision(self):
        self.alive = True
        for wall in self.walls:
            x1, y1, x2, y2 = wall[0], wall[1], wall[2], wall[3]
            x, y, m = None, None, None
            if(x2-x1 == 0):
                x = x1
                y = self.position[1]
            elif(y2-y1 == 0):
                x = self.position[0]
                y = y1
            else:
                m = (y2-y1)/(x2-x1)
                x = (y2 - y1 + m * x1 + x2 / m) / (m + 1.0 / m)
                y = y1 + m * (x - x1)
            d = int(math.sqrt(
                math.pow(x - self.position[0], 2) + math.pow(y - self.position[1], 2)))

            if(d < self.radius and
               (x-x1)*(y2-y1) == (x2-x1)*(y-y1)
                    and min(x1, x2) <= x
                    and min(y1, y2) <= y
                    and x <= max(x1, x2)
                    and y <= max(y1, y2)):
                self.alive = False
                break


def loadfiledata(file):
    f = open(file, 'r')
    rawdatas = f.readlines()
    f.close()
    return rawdatas


def loadfour(file):
    rawdatas = loadfiledata(file)
    fourdatas = []
    four_front_disdata = []
    four_right_disdata = []
    four_left_disdata = []
    four_steering_wheel_angle = []
    tmpX = []
    tmpY = []
    for i in range(0, len(rawdatas)):
        data = tuple(map(float, rawdatas[i].split(' ')))
        fourdatas.append(data)
        four_front_disdata.append(data[0])
        four_right_disdata.append(data[1])
        four_left_disdata.append(data[2])
        four_steering_wheel_angle.append(data[3])
        tmpX.append(data[:-1])
        tmpY.append(data[-1])
    tmpY = np.array(tmpY)
    tmpX = np.array(tmpX)
    return four_front_disdata, four_right_disdata, four_left_disdata, four_steering_wheel_angle, fourdatas, tmpX, tmpY


def loadroad(file):
    rawdatas = loadfiledata(file)
    start_position = [int(el) for el in rawdatas[0].split(',')]
    finish_line_UL = [int(el) for el in rawdatas[1].split(',')]
    finish_line_LR = [int(el) for el in rawdatas[2].split(',')]
    path_datas = rawdatas[3:]
    path_data = []
    for i in range(0, len(path_datas)):
        data = tuple(map(float, path_datas[i].split(',')))
        if(i == 0):
            path_data.append((Path.MOVETO, data))
        elif(i == len(path_datas)-1):
            path_data.append((Path.CLOSEPOLY, data))
        else:
            path_data.append((Path.LINETO, data))
    return path_data, start_position, finish_line_UL, finish_line_LR


def openfile(filetype):
    initialdir = os.getcwd()
    filename = tkinter.filedialog.askopenfilename(
        initialdir=initialdir, title="Select file", filetypes=[("Text Files", "*.txt")])
    if filename:
        if filetype == 'road':
            road_file_text.set('讀取軌道 : ' + os.path.basename(str(filename)))
            # loadroad(filename)
        elif filetype == 'fourD':
            fourD_file_text.set('訓練資料4D : ' + os.path.basename(str(filename)))
        elif filetype == 'sixD':
            sixD_file_text.set('訓練資料6D : ' + os.path.basename(str(filename)))


def run_sim(N, start_position, verts, fourdatas):
    cars = []
    for i in range(N):
        cars.append(Car([start_position[0], start_position[1]],
                        start_position[2], verts, fourdatas))
    for car in cars:
        fig, ax = plt.subplots()
        # plot the canvas
        canvs = FigureCanvasTkAgg(fig, root)
        canvs.get_tk_widget().grid(row=0, column=3, rowspan=12, columnspan=2,
                                   sticky=tkinter.W+tkinter.E+tkinter.N+tkinter.S, padx=5, pady=5)
        ax.add_patch(PathPatch(Path(verts, codes),
                               facecolor='y', alpha=0.7))  # 畫道路
        ax.add_patch(patches.Rectangle((finish_line_UL[0], finish_line_LR[1]), abs(finish_line_UL[0]-finish_line_LR[0]), abs(
            finish_line_UL[1]-finish_line_LR[1]), edgecolor='black', facecolor='red', fill=True))  # 畫終點線

        ax.axis('equal')
        car.draw(ax)
        car.update(fig)
        while True:
            if(car.alive == False or car.arrival):
                break
            root.update_idletasks()
            root.update()
        plt.close()


def training(N, start_position, verts, fourdatas):

    run_sim(N, start_position, verts, fourdatas)


root = tkinter.Tk()
root.title("電腦模擬車")

road_file_text = tkinter.StringVar()
road_file_ap = os.getcwd()+'/軌道座標點.txt'
road_file_text.set('讀取軌道 : ' + os.path.basename(road_file_ap))
tkinter.Label(root, textvariable=road_file_text).grid(row=0, sticky=tkinter.W)
tkinter.Button(root, text='選擇檔案', command=lambda: openfile(
    'road')).grid(row=0, column=1, sticky=tkinter.W)

fourD_file_text = tkinter.StringVar()
fourD_file_ap = os.getcwd()+'/train4dAll.txt'
fourD_file_text.set('訓練資料4D : ' + os.path.basename(fourD_file_ap))
tkinter.Label(root, textvariable=fourD_file_text).grid(row=1, sticky=tkinter.W)
tkinter.Button(root, text='選擇檔案', command=lambda: openfile(
    'fourD')).grid(row=1, column=1, sticky=tkinter.W)

sixD_file_text = tkinter.StringVar()
sixD_file_ap = os.getcwd()+'/train6dAll.txt'
sixD_file_text.set('訓練資料6D : ' + os.path.basename(sixD_file_ap))
tkinter.Label(root, textvariable=sixD_file_text).grid(row=2, sticky=tkinter.W)
tkinter.Button(root, text='選擇檔案', command=lambda: openfile(
    'sixD')).grid(row=2, column=1, sticky=tkinter.W)

path_data, start_position, finish_line_UL, finish_line_LR = loadroad(
    road_file_ap)
ffd, frd, fld, fswa, fourdatas, tmpX, tmpY = loadfour(fourD_file_ap)
codes, verts = zip(*path_data)
N = 1

tkinter.Button(root, text='開始', command=lambda: training(
    N, start_position, verts, fourdatas)).grid(row=3, column=1, sticky=tkinter.W)
root.mainloop()
