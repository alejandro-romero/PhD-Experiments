# class TracesMemory(object):
#     """ Class that represents a Memory of Traces
#
#     A trace is a list of episodes with an assigned value (expected reward)
#     which are stored together.
#     It distinguish between Positive Traces (named Traces), Negative Traces
#     (named AntiTraces) and WeakPositive Traces (named WeakTraces)
#
#     This class implements different methods to get/set the different traces
#     lists, get their contents and add/remove traces.
#     """
#
#     def __init__(self):
#         self.tracesList = []
#         self.antiTracesList = []
#         self.weakTracesList = []
#         # self.listsRemoved = []
#         # self.weaksRemoved = []
#         # self.traceListTime = []
#         ### Prueba FIFO
#         self.maxSize = 1200#750  # 25 traces * 30 points
#         ###
#
#     def addTraces(self, traces):
#         ### Prueba FIFO
#         if self.isFull(self.tracesList):
#             self.tracesList.pop(0)
#         ###
#         self.tracesList.append(traces)
#
#     def addAntiTraces(self, traces):
#         ### Prueba FIFO
#         if self.isFull(self.antiTracesList):
#             self.antiTracesList.pop(0)
#         ###
#         self.antiTracesList.append(traces)
#
#     def addWeakTraces(self, traces):
#         ### Prueba FIFO
#         if self.isFull(self.weakTracesList):
#             self.weakTracesList.pop(0)
#         ###
#         self.weakTracesList.append(traces)
#
#     def getTracesList(self):
#         return self.tracesList
#
#     def getAntiTracesList(self):
#         return self.antiTracesList
#
#     def getWeakTracesList(self):
#         return self.weakTracesList
#
#     ### Prueba FIFO
#     def getSize(self, list):
#         cont = 0
#         for i in range(len(list)):
#             cont += (len(list[i]))
#         return cont
#
#     def isFull(self, list):
#         return self.getSize(list) >= self.maxSize
#
#     def setMaxSize(self, maxSize):
#         self.maxSize = maxSize
#     ###
#
#     # def getListsRemoved(self):
#     #     return self.listsRemoved
#
#     # def setListsRemoved(self, listsRemoved):
#     #     self.listsRemoved = listsRemoved
#
#     # def addListsRemoved(self, listsToBeAdded):
#     #     self.listsRemoved.extend(listsToBeAdded)
#     #
#     # def getWeaksRemoved(self):
#     #     return self.weaksRemoved
#
#         # def setWeakTracesList(self, weakTracesList):
#         #     self.weakTracesList = weakTracesList
#
#     # def addWeaksRemoved(self, weakTracesList):
#     #     self.weaksRemoved.extend(weakTracesList)
class TracesMemory(object):
    """ Class that represents a Memory of Traces

    A trace is a list of episodes with an assigned value (expected reward)
    which are stored together.
    It distinguish between Positive Traces (named Traces), Negative Traces
    (named AntiTraces) and WeakPositive Traces (named WeakTraces)

    This class implements different methods to get/set the different traces
    lists, get their contents and add/remove traces.
    """

    def __init__(self):
        self.tracesList = []
        self.antiTracesList = []
        self.weakTracesList = []
        # self.listsRemoved = []
        # self.weaksRemoved = []
        # self.traceListTime = []

        self.IgTracesList = []

    def addIgTraces(self, traces):
        self.IgTracesList.append(traces)

    def getIgTracesList(self):
        return self.IgTracesList

    def addTraces(self, traces):
        self.tracesList.append(traces)

    def addAntiTraces(self, traces):
        self.antiTracesList.append(traces)

    def addWeakTraces(self, traces):
        self.weakTracesList.append(traces)

    def getTracesList(self):
        return self.tracesList

    def getAntiTracesList(self):
        return self.antiTracesList

    def getWeakTracesList(self):
        return self.weakTracesList

        # def getListsRemoved(self):
        #     return self.listsRemoved

        # def setListsRemoved(self, listsRemoved):
        #     self.listsRemoved = listsRemoved

        # def addListsRemoved(self, listsToBeAdded):
        #     self.listsRemoved.extend(listsToBeAdded)
        #
        # def getWeaksRemoved(self):
        #     return self.weaksRemoved

        # def setWeakTracesList(self, weakTracesList):
        #     self.weakTracesList = weakTracesList

        # def addWeaksRemoved(self, weakTracesList):
        #     self.weaksRemoved.extend(weakTracesList)