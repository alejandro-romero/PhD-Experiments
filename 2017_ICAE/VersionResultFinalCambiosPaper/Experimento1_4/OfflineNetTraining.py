from MDBCore import *
from BackProp import *
from mpl_toolkits.mplot3d import Axes3D
import random


def getDistance((x1, y1), (x2, y2)):
    """Return the distance between two points"""
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


a = MDBCore()
a.loadData()
# red = mlp

plt.close('all')

# Network input and output
in_data = []
out_data = []
for i in range(len(a.TracesMemoryVF.getTracesList())):
    for j in range(len(a.TracesMemoryVF.getTracesList()[i])):
        in_data.append(a.TracesMemoryVF.getTracesList()[i][j][0])
        out_data.append(a.TracesMemoryVF.getTracesList()[i][j][-1])
# for i in range(40):
#     for j in range(len(a.TracesMemoryVF.getTracesList()[-i])):
#         in_data.append(a.TracesMemoryVF.getTracesList()[-i][j][0])
#         out_data.append(a.TracesMemoryVF.getTracesList()[-i][j][-1])

# Normalise inputs
in_data = np.asarray(in_data)

# Trabajo en el intervalo 0-1300 (maximo valor sensor)
max_value = 0
for i in range(len(in_data)):
    if max(in_data[i]) > max_value:
        max_value = max(in_data[i])

in_data /= np.ceil(max_value)  # 1300  # Divido entre valor maximo para normalizar entre 0 y 1

# Draw input and output points
pl.figure()
pl.plot(in_data, out_data, 'o')
pl.grid()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(out_data)):
#     pl.scatter(in_data[:, 0][i], in_data[:, 1][i], out_data[i], marker='o')
# plt.xlabel('Point number')
# plt.ylabel('Value')
# pl.grid()

# Data vector
data = []
for i in range(len(in_data)):
    data.append((in_data[i][0], in_data[i][1], in_data[i][2], out_data[i]))
data = np.asarray(data)

# Mix data to train the network
random.shuffle(data)

# input = data[:, 0:2 + 1]
# output = data[:, 2 + 1]
input = data[:, 0:len(data[0]) - 1]
output = data[:, len(data[0]) - 1]

# input = input.reshape((input.shape[0], 1 + 1))
# output = output.reshape((input.shape[0], 1))
input = input.reshape((input.shape[0], len(data[0]) - 1))
output = output.reshape((input.shape[0], 1))

# # Input data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(input)):  # range(number):
#     if output[i] <= 0.2:
#         color = 'red'
#     elif output[i] <= 0.4:
#         color = 'orange'
#     elif output[i] <= 0.6:
#         color = 'green'
#     elif output[i] <= 0.8:
#         color = 'cyan'
#     elif output[i] <= 1.0:
#         color = 'blue'
#     else:
#         color = 'white'
#     ax.scatter(input[i][0], input[i][1], input[i][2], c=color, marker='x', linewidth=0.5)  # para mapa continuo sin s y sin linewidth
# ax.set_xlabel('dCR (norm.)', size=15.0)
# ax.set_ylabel('dCB (norm.)', size=15.0)
# ax.set_zlabel('dCX (norm.)', size=15.0)
# ###
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(input)):  # range(number):
#     if output[i] <= 0.2:
#         color = 'red'
#     elif output[i] <= 0.4:
#         color = 'orange'
#     elif output[i] <= 0.6:
#         color = 'green'
#     elif output[i] <= 0.8:
#         color = 'cyan'
#     elif output[i] <= 1.0:
#         color = 'blue'
#     else:
#         color = 'white'
#     ax.scatter(input[i][0], input[i][1], output[i], c=color, marker='x', linewidth=0.5)  # para mapa continuo sin s y sin linewidth
# ax.set_xlabel('dCR (norm.)', size=15.0)
# ax.set_ylabel('dCB (norm.)', size=15.0)
# ax.set_zlabel('Utility (a.u.)', size=15.0)
# ###
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(input)):  # range(number):
#     if output[i] <= 0.2:
#         color = 'red'
#     elif output[i] <= 0.4:
#         color = 'orange'
#     elif output[i] <= 0.6:
#         color = 'green'
#     elif output[i] <= 0.8:
#         color = 'cyan'
#     elif output[i] <= 1.0:
#         color = 'blue'
#     else:
#         color = 'white'
#     ax.scatter(input[i][0], input[i][2], output[i], c=color, marker='x', linewidth=0.5)  # para mapa continuo sin s y sin linewidth
# ax.set_xlabel('dCR (norm.)', size=15.0)
# ax.set_ylabel('dCX (norm.)', size=15.0)
# ax.set_zlabel('Utility (a.u.)', size=15.0)
# ###
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(input)):  # range(number):
#     if output[i] <= 0.2:
#         color = 'red'
#     elif output[i] <= 0.4:
#         color = 'orange'
#     elif output[i] <= 0.6:
#         color = 'green'
#     elif output[i] <= 0.8:
#         color = 'cyan'
#     elif output[i] <= 1.0:
#         color = 'blue'
#     else:
#         color = 'white'
#     ax.scatter(input[i][1], input[i][2], output[i], c=color, marker='x', linewidth=0.5)  # para mapa continuo sin s y sin linewidth
# ax.set_xlabel('dCB (norm.)', size=15.0)
# ax.set_ylabel('dCX (norm.)', size=15.0)
# ax.set_zlabel('Utility (a.u.)', size=15.0)
# ####


train = input[0::2, :]
test = input[1::4, :]
valid = input[3::4, :]
traintarget = output[0::2, :]
testtarget = output[1::4, :]
validtarget = output[3::4, :]

# Train net
plt.figure()
net = mlp(train, traintarget, 6, outtype='linear')
net.mlptrain(train, traintarget, 0.25, 101)
net.earlystopping(train, traintarget, valid, validtarget, 0.25)
test = np.concatenate((test, -np.ones((np.shape(test)[0], 1))), axis=1)
outputs = net.mlpfwd(test)
print 0.5 * sum((outputs - testtarget) ** 2)

###
train = input[0::2, :]
test = input[1::4, :]
valid = input[3::4, :]
traintarget = output[0::2, :]
testtarget = output[1::4, :]
validtarget = output[3::4, :]
# Train net
plt.figure()
net = mlp(train, traintarget, 10, outtype='linear')
net.mlptrain(train, traintarget, 0.25, 101)
net.earlystopping(train, traintarget, valid, validtarget, 0.25)
test = np.concatenate((test, -np.ones((np.shape(test)[0], 1))), axis=1)
outputs2 = net.mlpfwd(test)
print 0.5 * sum((outputs - testtarget) ** 2)
###
train = input[0::2, :]
test = input[1::4, :]
valid = input[3::4, :]
traintarget = output[0::2, :]
testtarget = output[1::4, :]
validtarget = output[3::4, :]
# Train net
plt.figure()
net = mlp(train, traintarget, 15, outtype='linear')
net.mlptrain(train, traintarget, 0.25, 101)
net.earlystopping(train, traintarget, valid, validtarget, 0.25)
test = np.concatenate((test, -np.ones((np.shape(test)[0], 1))), axis=1)
outputs3 = net.mlpfwd(test)
print 0.5 * sum((outputs - testtarget) ** 2)
###
# Draw output results
# plt.figure()
# plt.title('5 hidden nodes')
# plt.plot(outputs, 'x', label='output network')
# plt.plot(testtarget, 'o', label='output')
# plt.legend(loc='best', fontsize='medium')

# Draw output results sorted
comb = np.hstack((testtarget, outputs))
comb_sort = sorted(comb, key=lambda x: x[0])
a = np.array(comb_sort)

comb2 = np.hstack((testtarget, outputs2))
comb_sort2 = sorted(comb2, key=lambda x: x[0])
a2 = np.array(comb_sort2)

comb3 = np.hstack((testtarget, outputs3))
comb_sort3 = sorted(comb3, key=lambda x: x[0])
a3 = np.array(comb_sort3)

plt.figure()
# plt.title('5 hidden nodes', size=15.0)
plt.plot(a[:, 0], 'o', c='yellow', label='Target output')
plt.plot(a[:, 1], 'x', c='blue', label='Output network 6 hidden nodes')
plt.plot(a2[:, 1], 'o', c='red', fillstyle='none', label='Output network 10 hidden nodes')
plt.plot(a3[:, 1], '^', c='green', fillstyle='none', label='Output network 15 hidden nodes')
plt.legend(loc='best', fontsize='medium', numpoints=1)
plt.xlabel('Points', size=15.0)
plt.ylabel('Utility', size=15.0)
plt.grid()

# Representacion de error entre salidas y tambien entradas
error = abs(testtarget - outputs)
comb2 = np.hstack((error, test))
comb_sort2 = sorted(comb2, key=lambda x: x[0])
b = np.array(comb_sort2)

plt.figure()
plt.title('5 hidden nodes')
plt.plot(b[:, 0], 'x', label='error', color='black')
plt.plot(b[:, 1], '.', label='dCR', color='red')
plt.plot(b[:, 2], '.', label='dCB', color='green')
plt.plot(b[:, 3], '.', label='dCX', color='blue')
plt.legend(loc='best', fontsize='medium')
plt.grid()
######
# comb3 = np.hstack((error, test[:,0]))
comb_sort3 = sorted(comb2, key=lambda x: x[1])
c = np.array(comb_sort3)

plt.figure()
plt.title('5 hidden nodes')
plt.plot(c[:, 0], 'x', label='error', color='black')
plt.plot(c[:, 1], '.', label='dCR', color='red')
plt.legend(loc='best', fontsize='medium')
plt.grid()
######
# comb4 = np.hstack((error, test[:,1]))
comb_sort4 = sorted(comb2, key=lambda x: x[2])
d = np.array(comb_sort4)

plt.figure()
plt.title('5 hidden nodes')
plt.plot(d[:, 0], 'x', label='error', color='black')
plt.plot(d[:, 2], '.', label='dCB', color='green')
plt.legend(loc='best', fontsize='medium')
plt.grid()
######
# comb5 = np.hstack((error, test[:,2]))
comb_sort5 = sorted(comb2, key=lambda x: x[3])
e = np.array(comb_sort5)

plt.figure()
plt.title('5 hidden nodes')
plt.plot(e[:, 0], 'x', label='error', color='black')
plt.plot(e[:, 3], '.', label='dCX', color='blue')
plt.legend(loc='best', fontsize='medium')
plt.grid()



# import pickle
# f = open('DataTrainNet.pckl', 'rb')
# train=pickle.load(f)
# test=pickle.load(f)
# valid=pickle.load(f)
# traintarget=pickle.load(f)
# testtarget=pickle.load(f)
# validtarget=pickle.load(f)
# f.close()


# Evaluate individual points
# net.mlpfwd(test[-6].reshape(1, 3))

# plt.figure()
# dist = 0.01
# for i in range(len(outputs)):
#     plt.plot(i, outputs[i], 'x', color='blue')
#     plt.plot(i, testtarget[i], 'o', color='red')
#     for j in range(len(train)):
#         if getDistance(tuple(test[i, :2]), tuple(train[j])) < dist:
#             plt.plot(i, traintarget[j], 'o', color='red')
# plt.xlabel('Point number')
# plt.ylabel('Value')
# plt.grid()

# VER DISPERSION PUNTOS
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(in_data[:,0], in_data[:,1], out_data, linewidth=0.2, antialiased=True)
# plt.show()
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(test[:,0], test[:,1], outputs.reshape(229,), linewidth=0.2, antialiased=False, color='red')
# plt.show()


# Otra opcion
# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot a basic wireframe.
# ax.plot_wireframe(in_data[:,0], in_data[:,1], out_data, rstride=10, cstride=10)
# plt.show()

# Puntos 3D
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(in_data[:,0], in_data[:,1], out_data)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test[:,0], test[:,1], outputs.reshape(229,), color='red')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(in_data[:, 0], in_data[:, 1], out_data)
ax.scatter(test[:, 0], test[:, 1], outputs.reshape(outputs.shape[0], ), color='red')
ax.set_xlabel('Sensor 1')
ax.set_ylabel('Sensor 2')
ax.set_zlabel('Value')
plt.show()

# for i in range(len(outputs)):
#     plt.plot(i, outputs[i], 'x', color='red')



# Figura 3D salidas red
r = np.arange(0, 1.05, 0.025)
sens1 = []
sens2 = []
out_net = []
for i in r:
    for j in r:
        if j + i > 0.74:  # 1000/1300 ~= 0.76923...
            sens1.append(i)
            sens2.append(j)
            point = (i, j)
            point = np.asarray(point + (-1,))
            out_net.append(net.mlpfwd(point.reshape(1, 3)))
# Ploteo
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sens1, sens2, out_net)
ax.set_xlabel('db1 (norm.)')
ax.set_ylabel('db2 (norm.)')
ax.set_zlabel('Value (a.u.)')
plt.show()
ax.scatter(test[:, 0], test[:, 1], outputs.reshape(outputs.shape[0], ), color='red')
ax.scatter(in_data[:, 0], in_data[:, 1], out_data, color='green')

########
# # VIDEOS ENSTA - MODIFICACION LOGS
# f=open('logENSTA2.log',"r")
# lines=f.readlines()
# iterations=[]
# result=[]
# time=[]
# for x in lines:
#     time.append(x.split(' ')[2])
#     iterations.append(x.split(' ')[3])
#     result.append(x.split(' ')[26:28])
# f.close()
#
#
# file = open('logENSTA2_mod.txt', 'w')
# file.write('Time stamp'+' '+'Iteration'+' '+'Robobo_action'+ ' ' + 'Baxter_action'+'\n')
# for i in range(len(iterations)):
#     if i == 0:
#         file.write(str(time[i]) + ' ' + str(iterations[i]) + ' ' + str(result[i]) + '\n')
#     elif iterations[i] != iterations[i-1]:
#         file.write(str(time[i]) + ' ' + str(iterations[i]) + ' ' + str(result[i]) + '\n')
# file.close()


# # VIDEOS ENSTA - MODIFICACION LOGS
# f=open('mdb_motiv_rob-1_5.log',"r")
# lines=f.readlines()
# iterations=[]
# result=[]
# time=[]
# for x in lines:
#     # time.append(x.split(' ')[2])
#     iterations.append(x)
#     # result.append(x.split(' ')[26:28])
# f.close()
# file = open('logENSTA1_5.txt', 'w')
# for i in range(len(iterations)):
#     if i==0 or i % 2== 0:
#         file.write(str(iterations[i]) + '\n')
# file.close()



#### VFs

# # VF 1
# plt.figure()
# # Traces VF
# plt.title('Traces VF 1', fontsize=15.0)
# for i in (0,2,4,58,59,71,72,76):  # for i in range(25,52):
#     Trace = a.TracesMemoryVF.getTracesList()[-i]
#     x = []
#     y = []
#     for i in range(len(Trace)):
#         x.append(Trace[i][0][0])
#         y.append(Trace[i][0][1])
#     # plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
#     plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
#     plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
#     plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
# plt.xlabel('db1 (m)', fontsize=15.0)
# plt.ylabel('db2 (m)', fontsize=15.0)
# x = []
# y = []
# for i in range(0, 1000, 10):
#     x.append(i)
#     x.append(0)
#     y.append(0)
#     y.append(i)
#     plt.plot(x, y, marker='.', color='grey')
#     x = []
#     y = []
# plt.xlim(0, 1200)
# plt.ylim(0, 1200)
# plt.xticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
# plt.yticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
# plt.grid()
#
# # VF 2
# plt.figure()
# # Traces VF
# plt.title('Traces VF 2', fontsize=15.0)
# for i in (74,78,40,16,39,54,31,8,27):  # for i in range(25,52):
#     Trace = a.TracesMemoryVF.getTracesList()[-i]
#     x = []
#     y = []
#     for i in range(len(Trace)):
#         x.append(Trace[i][0][0])
#         y.append(Trace[i][0][1])
#     # plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
#     plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
#     plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
#     plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
# plt.xlabel('db1 (m)', fontsize=15.0)
# plt.ylabel('db2 (m)', fontsize=15.0)
# x = []
# y = []
# for i in range(0, 1000, 10):
#     x.append(i)
#     x.append(0)
#     y.append(0)
#     y.append(i)
#     plt.plot(x, y, marker='.', color='grey')
#     x = []
#     y = []
# plt.xlim(0, 1200)
# plt.ylim(0, 1200)
# plt.xticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
# plt.yticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
# plt.grid()
#
# # VF 3
# plt.figure()
# # Traces VF
# plt.title('Traces VF 3', fontsize=15.0)
# for i in (9,47,41,25,15,19,55,65,79,12,32,70,49,38,48,43):  # for i in range(25,52):
#     Trace = a.TracesMemoryVF.getTracesList()[-i]
#     x = []
#     y = []
#     for i in range(len(Trace)):
#         x.append(Trace[i][0][0])
#         y.append(Trace[i][0][1])
#     # plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
#     plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
#     plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
#     plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
# plt.xlabel('db1 (m)', fontsize=15.0)
# plt.ylabel('db2 (m)', fontsize=15.0)
# x = []
# y = []
# for i in range(0, 1000, 10):
#     x.append(i)
#     x.append(0)
#     y.append(0)
#     y.append(i)
#     plt.plot(x, y, marker='.', color='grey')
#     x = []
#     y = []
# plt.xlim(0, 1200)
# plt.ylim(0, 1200)
# plt.xticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
# plt.yticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
# plt.grid()

# # SUR
# plt.figure()
# plt.title('Traces SUR', fontsize=15.0)
# for i in (0,3,4,29,23,24):  # for i in range(25,52):
#     Trace = a.TracesMemoryVF.getTracesList()[-i]
#     x = []
#     y = []
#     for i in range(len(Trace)):
#         x.append(Trace[i][0][0])
#         y.append(Trace[i][0][1])
#     # plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
#     plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
#     plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
#     col = [i / 30.0, 1 - i / 30.0, 0.5]
#     plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
# plt.xlabel('db1 (m)', fontsize=15.0)
# plt.ylabel('db2 (m)', fontsize=15.0)
# x = []
# y = []
# for i in range(0, 1000, 10):
#     x.append(i)
#     x.append(0)
#     y.append(0)
#     y.append(i)
#     plt.plot(x, y, marker='.', color='grey')
#     x = []
#     y = []
# plt.xlim(0, 1200)
# plt.ylim(0, 1200)
# plt.grid()

# VF 4
plt.figure()
# Traces VF
plt.title('Traces VF 3', fontsize=15.0)
for i in (9, 25, 19, 55, 65, 32, 49, 43, 38, 48, 47, 15, 70):  # for i in range(25,52):
    # plt.figure()
    # plt.title(str(i))
    Trace = a.TracesMemoryVF.getTracesList()[-i]
    x = []
    y = []
    for i in range(len(Trace)):
        x.append(Trace[i][0][0])
        y.append(Trace[i][0][1])
    # plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
    plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
    plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
    plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
plt.xlabel('db1 (m)', fontsize=15.0)
plt.ylabel('db2 (m)', fontsize=15.0)

x = []
y = []
for i in range(0, 1000, 10):
    x.append(i)
    x.append(0)
    y.append(0)
    y.append(i)
    plt.plot(x, y, marker='.', color='grey')
    x = []
    y = []
plt.xlim(0, 1200)
plt.ylim(0, 1200)
plt.xticks([0, 200, 400, 600, 800, 1000, 1200], ['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
plt.yticks([0, 200, 400, 600, 800, 1000, 1200], ['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
plt.grid()
