import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from TracesMemory import *


class DistancesCertainty(object):
    """ This class represents a mathematical model that defines the
    implementation used for the creation of certainty maps. 

    Its aim is to obtain the certainty value for a point 'p' given.

    """

    def __init__(self):

        self.TracesMemory = TracesMemory()
        self.Nt = 0  # Number of traces: Nt=Nht+Cf*Nst: sum of p-traces and Cf times w-traces
        self.n_antitraces = 0
        self.numberOfGoalsWithoutAntiTraces = 0  # To check when a correlation has become established
        self.Cf = 0.7  # Reliability factor

        # Establecer estos limites antes de realizar los ejemplos en funcion de las distacias y el numero de sensores
        self.Linf = (0.0, 0.0)  # It should be an array with the inferior limits of the different sensors
        self.Lsup = (1373.0, 1373.0)  # It should be an array with the superior limits of the different sensors

        self.Nt_factor = 4.0#6.0
        self.K = pow(0.05, 1.0 / (self.Nt_factor - 1.0))
        self.ce = 1
        self.M = 50

        self.cp = 1.0  # 0.6  # Stability factor
        self.cw = 0.4  # 0.3  # Weighting factor for w-traces
        self.ca = 1.0  # Weighting factor for n-traces

        self.epsilon = 100

        self.percentile = 100  # q-th percentile

        self.tracesMinDistancesMap = ()
        self.weakTracesMinDistancesMap = ()
        self.antiTracesMinDistancesMap = ()

        self.plotMatrix = []
        # self.figure = plt.figure()
        plt.ion()
        # self.figure.canvas.set_window_title('PRUEBA')

    def getMinDistancesMap(self, T):
        """Return the set of the minimum distances for all the points in T.
         T is the set of trace points (episodes) used to define the certainty map."""
        D = [[None] * len(T) for i in range(len(T[0]))]
        N_GR = 99999
        for k in range(len(T)):
            for i in range(len(T[0])):
                d_pos = N_GR
                d_neg = d_pos
                for j in range(len(T)):
                    if k != j:
                        d = T[k][i] - T[j][i]
                        if d > 0:
                            d_pos = min(d_pos, d)
                        else:
                            d_neg = min(d_neg, -d)

                if d_pos > N_GR / 2:
                    d_pos = -1

                if d_neg > N_GR / 2:
                    d_neg = -1

                D[i][k] = max(d_pos, d_neg)

        return D

    def getPercentile(self, y, D):
        """Return the percentile 'y' over the set 'D'"""
        De = np.percentile(D, y, axis=1)
        De = De.tolist()

        return De

    def getDr(self, T):
        """Return the minimum distances to the sensor limits (distance
        from each of the n components of each episode contained in T)"""
        Dr = []
        for i in range(len(T[0])):
            dist_sup = abs(self.Lsup[i] - self.Linf[i])
            dist_inf = dist_sup

            for j in range(len(T)):
                dist_sup_tmp = abs(T[j][i] - self.Lsup[i])
                dist_inf_tmp = abs(T[j][i] - self.Linf[i])

                dist_sup = min(dist_sup, dist_sup_tmp)
                dist_inf = min(dist_inf, dist_inf_tmp)

            dist = max(dist_sup, dist_inf)
            Dr.append(dist)

        return Dr

    def get_h(self, T, p):
        """Return the distances between each of the n components of the trace points contained in T and any point p"""
        h = [[None] * len(T) for i in range(len(T[0]))]
        for i in range(len(T[0])):
            for j in range(len(T)):
                h[i][j] = abs(p[i] - T[j][i])

        return h

    def getHlim(self, MinDistancesMap, percentile, T, n_traces):
        """Return Hlim. the limit distances in the m dimensions from
        which traces quickly decrease their effect on the state space"""
        De = self.getPercentile(percentile, MinDistancesMap)
        Dr = self.getDr(T)

        Hlim = []
        for i in range(len(Dr)):
            if Dr[i] > De[i]:
                Hlim.append((De[i] + (Dr[i] - De[i]) * pow(self.K, n_traces - 1)) / 2.0)
            else:
                Hlim.append(De[i] / 2.0)

        # print "Hlim: ", Hlim
        return Hlim

    def get_hn(self, MinDistancesMap, percentile, T, n_traces, p):
        """Return the effective distances in the m dimensions between the trace points and any point p"""

        h = self.get_h(T, p)
        Hlim = self.getHlim(MinDistancesMap, percentile, T, n_traces)

        hn = [[None] * len(h) for i in range(len(h[0]))]
        for i in range(len(h)):
            for j in range(len(h[0])):
                if h[i][j] < self.ce * Hlim[i]:
                    hn[j][i] = h[i][j]
                else:
                    hn[j][i] = self.ce * Hlim[i] + (h[i][j] - self.ce * Hlim[i]) * self.M  # hn[j][i]=999999999#

        return hn, h

    def getWeight(self, MinDistancesMap, percentile, T, n_traces, p):
        """Return the weights of the trace points in any point p"""

        hn, h = self.get_hn(MinDistancesMap, percentile, T, n_traces, p)

        W = []
        for i in range(len(hn)):
            norm_value = []
            norm_value_aux = []
            for j in range(len(hn[0])):
                norm_value.append(hn[i][j] / (self.Lsup[j] - self.Linf[j]))
                norm_value_aux.append(h[j][i] / (self.Lsup[j] - self.Linf[j]))
            W.append(max(0, 1 - np.linalg.norm(norm_value)) / (np.linalg.norm(norm_value_aux) + 1 / self.epsilon))

        return W

    def getCertaintyValue(self, p):
        """Return the certainty value C for a point p combining the weights of
         p-traces(w_positive), n-traces(w_negative) and w-traces(w_weak)"""
        TracesTuple = self.TraceListToTuple(self.TracesMemory.getTracesList())
        AntiTracesTuple = self.TraceListToTuple(self.TracesMemory.getAntiTracesList())
        WeakTracesTuple = self.TraceListToTuple(self.TracesMemory.getWeakTracesList())

        if TracesTuple is ():
            w_positive = ()
        else:
            w_positive = self.getWeight(self.tracesMinDistancesMap, self.percentile, TracesTuple, self.Nt, p)

        if AntiTracesTuple is ():
            w_negative = ()
        else:
            w_negative = self.getWeight(self.antiTracesMinDistancesMap, self.percentile, AntiTracesTuple,
                                        self.n_antitraces, p)

        if WeakTracesTuple is ():
            w_weak = ()
        else:
            w_weak = self.getWeight(self.weakTracesMinDistancesMap, self.percentile, WeakTracesTuple, self.Nt, p)

        Sum = 0
        for i in range(len(w_positive)):
            Sum += w_positive[i]
        for i in range(len(w_weak)):
            Sum += w_weak[i]
            # Sum += self.cw * w_weak[i]
        for i in range(len(w_negative)):
            Sum -= w_negative[i]
            # Sum -= self.ca * w_negative[i]

        # if len(w_positive) > 0:
        #     W_pos = 1
        #     for i in range(len(w_positive)):
        #         W_pos *= (1 - w_positive[i])
        #     W_pos = 1-W_pos
        # else:
        #     W_pos = 0
        #
        # if len(w_weak) > 0:
        #     W_weak = 1
        #     for i in range(len(w_weak)):
        #         W_weak *= (1 - w_weak[i])
        #     W_weak = 1-W_weak
        # else:
        #     W_weak = 0
        #
        # if len(w_negative) > 0:
        #     W_neg = 1
        #     for i in range(len(w_negative)):
        #         W_neg *= (1 - w_negative[i])
        #     W_neg=1-W_neg
        # else:
        #     W_neg = 0
        #
        # Sum = W_pos + W_weak - W_neg

        # C = max(0, math.tanh(self.cp * Sum))
        # C = max(0, self.sigmoid(self.cp * Sum, 201))

        # C = ((math.tanh(self.cp * Sum)+1)/2.0)


        # C = max(0, np.sign(Sum)*math.pow(abs(math.tanh(self.cp * Sum)), 0.1))
        C = np.sign(Sum) * math.pow(abs(math.tanh(self.cp * Sum)), 0.1)

        return C

    def addTraces(self, newTrace, IntrinsicGuided):

        self.TracesMemory.addTraces(newTrace)
        self.Nt += 1
        # if not IntrinsicGuided:
        self.numberOfGoalsWithoutAntiTraces += 1

        # Update traces minimum distances map
        T = self.TraceListToTuple(self.TracesMemory.getTracesList())
        self.tracesMinDistancesMap = self.getMinDistancesMap(T)

        # Show certainty map
        # self.DrawPoints()
        # self.DrawCertaintyMap()
        # self.DrawTrace('p', newTrace)

        self.SaveTraceDraw(newTrace, 'p')

    def addWeakTraces(self, newTrace):

        self.TracesMemory.addWeakTraces(newTrace)
        self.Nt += self.Cf * 1

        # Update traces minimum distances map
        T = self.TraceListToTuple(self.TracesMemory.getWeakTracesList())
        self.weakTracesMinDistancesMap = self.getMinDistancesMap(T)

        # Show certainty map
        # self.DrawPoints()
        # self.DrawCertaintyMap()
        # self.DrawTrace('w', newTrace)

        self.SaveTraceDraw(newTrace, 'w')

    def addAntiTraces(self, newTrace, IntrinsicGuided):

        self.TracesMemory.addAntiTraces(newTrace)
        self.n_antitraces += 1
        # self.numberOfGoalsWithoutAntiTraces = 0
        if not IntrinsicGuided:
            self.numberOfGoalsWithoutAntiTraces = max(0, self.numberOfGoalsWithoutAntiTraces - 1)

        # Update traces minimum distances map
        T = self.TraceListToTuple(self.TracesMemory.getAntiTracesList())
        self.antiTracesMinDistancesMap = self.getMinDistancesMap(T)

        # Show certainty map
        # self.DrawPoints()
        # self.DrawTrace('n', newTrace)

        self.SaveTraceDraw(newTrace, 'n')

    def addIgTraces(self, newTrace, IntrinsicGuided):

        self.TracesMemory.addIgTraces(newTrace)

        # Show certainty map
        # self.DrawPoints()
        # self.DrawTrace('n', newTrace)

        self.SaveTraceDraw(newTrace, 'Ig')

    def getNumberOfGoalslWithoutAntiTraces(self):
        return self.numberOfGoalsWithoutAntiTraces

    def TraceListToTuple(self, TraceList):
        """ Transform a list into a tuple

        :param TraceList: a list of traces containing episodes (tuples)
        :return: a tuple of traces containing episodes (tuples)
        """
        TraceTuple = ()
        for i in range(len(TraceList)):
            TraceTuple += TraceList[i]

        return TraceTuple

    def SaveTraceDraw(self, Trace, type):
        """Save data in the matrix to plot"""
        for i in range(len(Trace)):
            addToMatrix = ()
            # Trace episode
            addToMatrix += Trace[i]
            # Type of trace: positive, negative or weak
            # if type == 'p':
            #     addToMatrix += (1, 0, 0)
            # elif type == 'n':
            #     addToMatrix += (0, 1, 0)
            # else:
            #     addToMatrix += (0, 0, 1)
            if type == 'p':
                addToMatrix += (1, 0, 0, 0)
            elif type == 'n':
                addToMatrix += (0, 1, 0, 0)
            elif type == 'w':
                addToMatrix += (0, 0, 1, 0)
            else:  # type =='Ig'
                addToMatrix += (0, 0, 0, 1)
            # # Slice in the 1st subspace
            # if Trace[i][1] < 250:
            #     addToMatrix += (1, 0, 0, 0)
            # elif Trace[i][1] < 750:
            #     addToMatrix += (0, 1, 0, 0)
            # elif Trace[i][1] < 1250:
            #     addToMatrix += (0, 0, 1, 0)
            # else:
            #     addToMatrix += (0, 0, 0, 1)
            # # Slice in the 2nd subspace
            # if Trace[i][0] < 250:
            #     addToMatrix += (1, 0, 0, 0)
            # elif Trace[i][0] < 750:
            #     addToMatrix += (0, 1, 0, 0)
            # elif Trace[i][0] < 1250:
            #     addToMatrix += (0, 0, 1, 0)
            # else:
            #     addToMatrix += (0, 0, 0, 1)
            self.plotMatrix.append(addToMatrix)

    def DrawTrace(self, type, Trace):#, subspace):
        """Draw a line that represents a trace and all its trace points

        :param type: a string that indicates if it is a p-trace (p), n-trace (n) or w-trace (w)
        :param Trace: A tuple of tuples that represents the trace with its episodes coordinates (x,y)
        """

        # Reorganize episode values to plot them
        x = []
        y = []
        for i in range(len(Trace)):
            x.append(Trace[i][0])
            y.append(Trace[i][1])

        # Set the color to differentiate the type of trace
        if type == 'w':
            col = 'green'  # 'b'
        elif type == 'p':
            col = 'black'  # 'black'
        elif type == 'n':
            col = 'grey'  # red'grey'
        elif type == 'Ig':
            col = 'orange'  # red'grey'

        plt.plot(x, y, marker='.', color=col, linewidth=5)  # 8

    def DrawPoints(self):
        """Draws n random (x,y) points and colours them according to they certainty value

        :param number: number of random points to generate (default 100 points)
        """

        # plt.figure()
        plt.clf()

        plt.axes(xlim=(self.Linf[0], self.Lsup[0]), ylim=(self.Linf[0], self.Lsup[0]))
        plt.xlabel('distance ball-balise1')
        plt.ylabel('distance ball-balise2')
        # a = range(self.Linf[0], self.Lsup[0], 43)  # 10 para mapa continuo
        a = np.arange(self.Linf[0], self.Lsup[0], 43)
        a = a.tolist()
        for i in range(len(a)):  # range(number):
            for j in range(len(a)):
                x = a[i]  # np.random.uniform(self.Linf[0], self.Lsup[0])
                y = a[j]  # np.random.uniform(self.Linf[0], self.Lsup[0])

                # Obtengo certeza para ese x,y
                certainty = max(0, self.getCertaintyValue((x, y)))

                # Establezco el color del punto en funcion del valor de certeza
                color = (math.pow(1 - certainty, 0.5), math.pow(certainty, 0.5), 0)

                # Dibujo el punto
                plt.scatter(x, y, c=color, marker='o', s=30, linewidth=0)  # para mapa continuo sin s y sin linewidth
        self.DrawTracesSubSpace()
        plt.draw()

    def DrawTracesSubSpace(self):#, subspace, slice):
        """Draw traces according to the subspace and the slice they belong to"""
        # DrawTraces
        for i in range(len(self.TracesMemory.getTracesList())):
            T = self.TracesMemory.getTracesList()[i]
            self.DrawTrace('p', T)#, subspace)
        # DrawWeakTraces
        for i in range(len(self.TracesMemory.getWeakTracesList())):
            T = self.TracesMemory.getWeakTracesList()[i]
            self.DrawTrace('w', T)#, subspace)
        # DrawAntiTraces
        for i in range(len(self.TracesMemory.getAntiTracesList())):
            T = self.TracesMemory.getAntiTracesList()[i]
            self.DrawTrace('n', T)#, subspace)

        # for i in range(len(self.TracesMemory.getIgTracesList())):
        #     T = self.TracesMemory.getIgTracesList()[i]
        #     self.DrawTrace('Ig', T)#, subspace)


    def DrawPointsPerfect(self):
        """Draws n random (x,y) points and colours them according to they certainty value

        :param number: number of random points to generate (default 100 points)
        """

        plt.figure()
        plt.axes(xlim=(self.Linf[0], self.Lsup[0]), ylim=(self.Linf[0], self.Lsup[0]))
        plt.xlabel('distance ball-robot')
        plt.ylabel('distance ball-box')
        a = range(self.Linf[0], self.Lsup[0], 30)  # 10 para mapa continuo
        for i in range(len(a)):  # range(number):
            for j in range(len(a)):
                x = a[i]  # np.random.uniform(self.Linf[0], self.Lsup[0])
                y = a[j]  # np.random.uniform(self.Linf[0], self.Lsup[0])

                # Obtengo certeza para ese x,y
                certainty = self.getCertaintyValue((x, y))

                # Establezco el color del punto en funcion del valor de certeza
                color = (math.pow(1 - certainty, 0.5), math.pow(certainty, 0.5), 0)

                # Dibujo el punto
                plt.scatter(x, y, c=color, marker='s')  # ,  linewidth=0)  #s=50, # para mapa continuo sin s y sin linewidth

    def DrawCertaintyMap(self):
        a = np.arange(self.Linf[1], self.Lsup[1], 40)
        a = a.tolist()
        certainty = [[None] * len(a) for i in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a)):
                p = a[i]
                q = a[j]
                certainty[i][j] = max(0,self.getCertaintyValue((p, q)))

        plt.figure()
        plt.axes(xlim=(self.Linf[0], self.Lsup[0]), ylim=(self.Linf[0], self.Lsup[0]))
        plt.xlabel('distance ball-robot')
        plt.ylabel('distance ball-box')

        x = a
        y = a

        # setup the 2D grid with Numpy
        x, y = np.meshgrid(x, y)
        # convert intensity (list of lists) to a numpy array for plotting
        certainty = np.array(certainty).transpose()
        # now just plug the data into pcolormesh, it's that easy!
        plt.pcolormesh(x, y, certainty, cmap='winter')
        plt.colorbar()  # need a colorbar to show the intensity scale
        plt.show()  # boom


    # def DrawTracesSubSpace(self, subspace, slice):
    #     """Draw traces according to the subspace and the slice they belong to"""
    #     # DrawTraces
    #     for i in range(len(self.TracesMemory.getTracesList())):
    #         T = self.TracesMemory.getTracesList()[i]
    #         self.DrawTrace('p', T, subspace)
    #     # DrawWeakTraces
    #     for i in range(len(self.TracesMemory.getWeakTracesList())):
    #         T = self.TracesMemory.getWeakTracesList()[i]
    #         self.DrawTrace('w', T, subspace)
    #     # DrawAntiTraces
    #     for i in range(len(self.TracesMemory.getAntiTracesList())):
    #         T = self.TracesMemory.getAntiTracesList()[i]
    #         self.DrawTrace('n', T, subspace)
    #
    # def MarkTracePoints(self, subspace, slice):
    #     """Highlight the points of the trace that belong to a determinate subspace and slice"""
    #     for i in range(len(self.plotMatrix)):
    #         # Points
    #         x, y, z = self.plotMatrix[i][0:3]  # Porque no incluye el ultimo elemento del rango
    #         # Color depending on the trace type
    #         if self.plotMatrix[i][3]:  # Positive trace
    #             color = 'black'
    #         elif self.plotMatrix[i][4]:  # Negative trace
    #             color = 'grey'
    #         elif self.plotMatrix[i][5]:  # Weak trace
    #             color = 'green'
    #
    #         if subspace == 1:
    #             if slice == 1 and self.plotMatrix[i][6]:
    #                 plt.scatter(x, z, c=color, marker='o', s=50, linewidth=1.0)
    #             elif slice == 2 and self.plotMatrix[i][7]:
    #                 plt.scatter(x, z, c=color, marker='o', s=50, linewidth=1.0)
    #             elif slice == 3 and self.plotMatrix[i][8]:
    #                 plt.scatter(x, z, c=color, marker='o', s=50, linewidth=1.0)
    #             elif slice == 4 and self.plotMatrix[i][9]:
    #                 plt.scatter(x, z, c=color, marker='o', s=50, linewidth=1.0)
    #         else:  # subspace == 2
    #             if slice == 1 and self.plotMatrix[i][10]:
    #                 plt.scatter(y, z, c=color, marker='o', s=50, linewidth=1.0)
    #             elif slice == 2 and self.plotMatrix[i][11]:
    #                 plt.scatter(y, z, c=color, marker='o', s=50, linewidth=1.0)
    #             elif slice == 3 and self.plotMatrix[i][12]:
    #                 plt.scatter(y, z, c=color, marker='o', s=50, linewidth=1.0)
    #             elif slice == 4 and self.plotMatrix[i][13]:
    #                 plt.scatter(y, z, c=color, marker='o', s=50, linewidth=1.0)

    def getCertaintyMatrix(self):

        a = np.arange(self.Linf[1], self.Lsup[1], 10)
        a = a.tolist()
        certaintyMatrix = np.zeros((len(a), len(a)))
        for i in range(len(a)):  # range(number):
            for j in range(len(a)):
                    x = a[i]  # np.random.uniform(self.Linf[0], self.Lsup[0])
                    y = a[j]
                    # Obtengo certeza para ese x,y
                    certainty = self.getCertaintyValue((x, y))
                    # Si hay certeza guardo un uno. En otro caso un cero
                    if certainty > 0.0:
                        certaintyMatrix[i][j] = 1
                    else:
                        certaintyMatrix[i][j] = 0
        return certaintyMatrix
