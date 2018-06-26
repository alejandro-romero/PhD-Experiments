from BackProp import *
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = open('leftArmMovement.txt', 'r')
data = [map(float,line.split()) for line in f]

in_data=[]
out_data=[]
for i in range(len(data)):
    out_data.append(data[i][3:])
    in_data.append(data[i][:3])

# Normalise inputs
in_data = np.asarray(in_data)

# Trabajo en el intervalo 0-1300 (maximo valor sensor)
# in_data /= 1300  # Divido entre valor maximo para normalizar entre 0 y 1

# Draw input and output points
plt.figure()
plt.plot(in_data, out_data, 'o')
plt.grid()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(out_data)):
#     pl.scatter(in_data[:, 0][i], in_data[:, 1][i], out_data[i], marker='o')
# plt.xlabel('Point number')
# plt.ylabel('Value')
# pl.grid()

# Data vector
data = []
for i in range(1500):# for i in range(len(in_data)):
    data.append((in_data[i][0], in_data[i][1], in_data[i][2], out_data[i][0], out_data[i][1]))
data = np.asarray(data)

# Mix data to train the network
random.shuffle(data)

input = data[:, 0:2 + 1]
output = data[:, 3:4 + 1]

input = input.reshape((input.shape[0], 2 + 1))
output = output.reshape((input.shape[0], 2))

train = input[0::2, :]
test = input[1::4, :]
valid = input[3::4, :]
traintarget = output[0::2, :]
testtarget = output[1::4, :]
validtarget = output[3::4, :]

# Train net
plt.figure()
net = mlp(train, traintarget, 5, outtype='linear')
net.mlptrain(train, traintarget, 0.25, 101)
net.earlystopping(train, traintarget, valid, validtarget, 0.25)
test = np.concatenate((test, -np.ones((np.shape(test)[0], 1))), axis=1)
outputs = net.mlpfwd(test)
print 0.5 * sum((outputs - testtarget) ** 2)

# Draw output results
plt.figure()
plt.title('5 hidden nodes')
# plt.plot(outputs[:,0], 'x', label='output network - distance_t+1')
plt.plot(outputs[:,1], 'x', label='output network - angle_t+1', markersize=5.0, color='black')
# plt.plot(testtarget[:,0], 'o', label='output - distance_t+1')
plt.plot(testtarget[:,1], 'o', label='output - angle_t+1')
plt.legend(loc='best', fontsize='medium')

plt.figure()
plt.title('5 hidden nodes')
plt.plot(outputs[:,0], 'x', label='output network - distance_t+1', markersize=5.0, color='black')
# plt.plot(outputs[:,1], 'x', label='output network - angle_t+1')
plt.plot(testtarget[:,0], 'o', label='output - distance_t+1', color='red')
# plt.plot(testtarget[:,1], 'o', label='output - angle_t+1')
plt.legend(loc='best', fontsize='medium')


# # Evaluate individual points
# net.mlpfwd(test[-6].reshape(1, 3))
#
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