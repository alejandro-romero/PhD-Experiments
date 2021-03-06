from DistancesCertainty import *


class Correlations(object):
    """Class that represents the Correlations module.
    This module identifies new correlations and contains the set of existing correlations.
    
    It contains the Correlation Evaluator that is an algorithm that has to be executed each time a trace is added
    to the Traces Memory and searches for possible correlations to be stored in the Traces Correlation Memory.
    
    It also has the Distances Certainty module that makes possible the creation of certainty maps using the traces 
    stored as positive-traces, negative-traces and weak-traces, which aim is to obtain the certainty value for a
    point p given
    """

    def __init__(self, rewardAssigner):
        self.n_sensor = 2  # Number of sensors. Useful to know how many possible correlations there are
        self.min_ep = 5  # Minimum number of episodes to consider the correlation possible
        self.same_values_accepted = 1  # Number of sensor admitted to be equal

        # Correlations Traces Memories and certainty evaluators
        self.S1_pos = DistancesCertainty()
        self.S1_neg = DistancesCertainty()
        self.S2_neg = DistancesCertainty()
        self.S2_pos = DistancesCertainty()

        self.corr_active = 0  # 1 - Sensor 1, 2 - Sensor 2, ... n- sensor n, 0 - no hay correlacion
        self.corr_type = ''  # 'pos' - Correlacion positiva, 'neg' - Correlacion negativa, '' - no hay correlacion
        self.corr_threshold = 0.1  # Threshold to know when to consider Extrinsic Motivation and when Intrinsic

        self.established = 0
        self.corr_established = 0
        self.corr_established_type = ''

        self.i_reward = rewardAssigner  # Valor que indica el indice de la correlacion encargada de
        # asignarle el reward. Si el indice es null, sera el escenario el encargado de hacerlo

        ##########
        self.i_reward_assigned = 0  # Valor para conocer si ya le ha sido asignada una correlacion asignadora de reward
        ##########

        self.Tb_max = 10#5  # Tb_max: Number of goals without antitraces needed to consider the correlation established

        self.figure = plt.figure()
        # self.figure.canvas.set_window_title('PRUEBA')

    def correlationEvaluator(self, Trace):
        """This method evaluates the possible correlations existing in a trace T and save them in the proper Correlation 
        Traces Memory Buffer
        
        Keyword arguments:
        Trace -- List of tuples, it is a List of episodes-sensorization(tuples)
        """
        if len(Trace) >= self.min_ep:
            for i in range(self.n_sensor):
                p_corr = 1  # Positive correlation
                n_corr = 1  # Negative correlation
                same_value = 0  # Number of times a sensor has the same value in two consecutive episodes
                for j in reversed(range(len(Trace) - self.min_ep, len(Trace))):
                    # No es necesario llegar al 0 porque estoy contemplandolo ya en el [j-1]
                    if p_corr:  # The case when positive correlation is active
                        if Trace[j][i] > Trace[j - 1][i]:
                            n_corr = 0  # Negative correlation is no longer possible for this sensor
                        elif Trace[j][i] < Trace[j - 1][i]:
                            p_corr = 0  # Positive correlation is no longer possible for this sensor
                        else:  # Trace[j][i]=Trace[j-1][i]
                            same_value += 1
                            if same_value > self.same_values_accepted:
                                n_corr = 0
                                p_corr = 0
                    elif n_corr:  # The case when negative correlation is active
                        if Trace[j][i] > Trace[j - 1][i]:
                            n_corr = 0  # Negative correlation is no longer possible for this sensor
                        elif Trace[j][i] < Trace[j - 1][i]:
                            p_corr = 0  # Positive correlation is no longer possible for this sensor
                        else:  # Trace[j][i]=Trace[j-1][i]
                            same_value += 1
                            if same_value > self.same_values_accepted:
                                n_corr = 0
                                p_corr = 0

                # If there is a correlation, save it in the pertinent correlation trace memory
                if p_corr:
                    self.addWeakTrace(Trace, i + 1, 'pos')
                    # if i == 0:
                    #     self.addWeakTrace(Trace, i + 1, 'pos')
                    # elif i == 1:
                    #     self.addWeakTrace(Trace, i + 1, 'pos')
                elif n_corr:
                    self.addWeakTrace(Trace, i + 1, 'neg')
                    # if i == 0:
                    #     self.addWeakTrace(Trace, i + 1, 'neg')
                    # elif i == 1:
                    #     self.addWeakTrace(Trace, i + 1, 'neg')

    def getActiveCorrelation(self, p):
        """
        # Este metodo despues ira dentro del motivation manager, y lo de correlaciones importado alli tambien (dentro de un modulo que sea modelos de utilidad)

        # Evaluo la certeza del nuevo punto en todas las correlaciones para ver si pertenece a alguna
        # Si es mayor que un umbral para alguna de ellas, considero la mayor y si hay empate, una al azar
        # Si es menor que el umbral, consireo la motivacion intrinseca        
        :param p: 
        :return: 
        """

        c1_pos = self.S1_pos.getCertaintyValue(p)
        c1_neg = self.S1_neg.getCertaintyValue(p)
        c2_pos = self.S2_pos.getCertaintyValue(p)
        c2_neg = self.S2_neg.getCertaintyValue(p)

        n_c1_pos = self.S1_pos.numberOfGoalsWithoutAntiTraces
        n_c1_neg = self.S1_neg.numberOfGoalsWithoutAntiTraces
        n_c2_pos = self.S2_pos.numberOfGoalsWithoutAntiTraces
        n_c2_neg = self.S2_neg.numberOfGoalsWithoutAntiTraces

        if self.established: ### Poido non estar de acordo en esto
            # j = (n_c1_pos, n_c1_neg, n_c2_pos, n_c2_neg).index(max(n_c1_pos, n_c1_neg, n_c2_pos, n_c2_neg))  # Save index of the correlated correlation
            # if self.corr_threshold > (c1_pos, c1_neg, c2_pos, c2_neg)[j]: # Si el umbral es mayor que el valor de certeza de la correlacion consolidada
            #     self.corr_active = 0
            #     self.corr_type = ''
            # else:
            #     i = (n_c1_pos, n_c1_neg, n_c2_pos, n_c2_neg).index(max(n_c1_pos, n_c1_neg, n_c2_pos, n_c2_neg)) # Redundante, ya podria usar el indice j de arriba
            #     if i < 2:
            #         self.corr_active = 1  # Sensor 1
            #     else:
            #         self.corr_active = 2  # Sensor 2
            #     if i % 2 == 0:  # Posicion par
            #         self.corr_type = 'pos'
            #     else:
            #         self.corr_type = 'neg'
            if self.corr_established == 1:
                if self.corr_established_type == 'pos':
                    j = 0
                else:
                    j = 1
            else: #elif self.corr_established == 2:
                if self.corr_established_type == 'pos':
                    j = 2
                else:
                    j = 3

            if self.corr_threshold > (c1_pos, c1_neg, c2_pos, c2_neg)[j]:  # Si el umbral es mayor que el valor de certeza de la correlacion consolidada
                self.corr_active = 0
                self.corr_type = ''
            else:
                self.corr_active = self.corr_established
                self.corr_type = self.corr_established_type

        else:
            if self.corr_threshold > max(c1_pos, c1_neg, c2_pos, c2_neg):
                self.corr_active = 0  # Al no haber correlacion activa doy por hecho que se usa la motivInt
                self.corr_type = ''
            else:
                # Guardo posicion valor maximo
                i = (c1_pos, c1_neg, c2_pos, c2_neg).index(max(c1_pos, c1_neg, c2_pos, c2_neg))
                if i < 2:
                    self.corr_active = 1  # Sensor 1
                else:
                    self.corr_active = 2  # Sensor 2
                if i % 2 == 0:  # Posicion par
                    self.corr_type = 'pos'
                else:
                    self.corr_type = 'neg'

        # certainty_value = max(c1_pos, c1_neg, c2_pos, c2_neg)

        return self.corr_active, self.corr_type  # , certainty_value

    def getCertainty(self, p):
        """This method provides the maximum certainty value of the correlations
        
        :param p: 
        :return: 
        """
        c1_pos = self.S1_pos.getCertaintyValue(p)
        c1_neg = self.S1_neg.getCertaintyValue(p)
        c2_pos = self.S2_pos.getCertaintyValue(p)
        c2_neg = self.S2_neg.getCertaintyValue(p)

        if self.established :
             # n_c1_pos = self.S1_pos.numberOfGoalsWithoutAntiTraces
             # n_c1_neg = self.S1_neg.numberOfGoalsWithoutAntiTraces
             # n_c2_pos = self.S2_pos.numberOfGoalsWithoutAntiTraces
             # n_c2_neg = self.S2_neg.numberOfGoalsWithoutAntiTraces
             #
             # j = (n_c1_pos, n_c1_neg, n_c2_pos, n_c2_neg).index(
             #     max(n_c1_pos, n_c1_neg, n_c2_pos, n_c2_neg))  # Save index of the correlated correlation
             #
             # certainty_value = (c1_pos, c1_neg, c2_pos, c2_neg)[j] # Certainty value of the correlated correlation
            if self.corr_established == 1:
                if self.corr_established_type == 'pos':
                    j = 0
                else:
                    j = 1
            else: #elif self.corr_established == 2:
                if self.corr_established_type == 'pos':
                    j = 2
                else:
                    j = 3
            certainty_value = (c1_pos, c1_neg, c2_pos, c2_neg)[j]
        else:
            certainty_value = max(c1_pos, c1_neg, c2_pos, c2_neg)

        return certainty_value

    def addTrace(self, Trace, sensor, corr_type, IntrinsicGuided=0):

        if not self.established:
            # Guardo solo hasta donde se cumple la correlacion
            for i in reversed(range(len(Trace))):
                if corr_type == 'neg':
                    if Trace[i][sensor - 1] >= Trace[i - 1][sensor - 1]:
                        break
                elif corr_type == 'pos':
                    if Trace[i][sensor - 1] <= Trace[i - 1][sensor - 1]:
                        break

            if sensor == 1:
                if corr_type == 'pos':
                    self.S1_pos.addTraces(Trace[i:], IntrinsicGuided)
                elif corr_type == 'neg':
                    self.S1_neg.addTraces(Trace[i:], IntrinsicGuided)
            elif sensor == 2:
                if corr_type == 'pos':
                    self.S2_pos.addTraces(Trace[i:], IntrinsicGuided)
                elif corr_type == 'neg':
                    self.S2_neg.addTraces(Trace[i:], IntrinsicGuided)

        # Check if the correlation is established (it could only happen after adding a trace)
        self.isCorrelationEstablished()

    def addAntiTrace(self, Trace, sensor, corr_type, IntrinsicGuided=0):
        # Filtro aqui para guardar los valores obtenidos con motivacion extrinseca
        # if not self.established:
            if sensor == 1:
                if corr_type == 'pos':
                    self.S1_pos.addAntiTraces(Trace, IntrinsicGuided)
                elif corr_type == 'neg':
                    self.S1_neg.addAntiTraces(Trace, IntrinsicGuided)
            elif sensor == 2:
                if corr_type == 'pos':
                    self.S2_pos.addAntiTraces(Trace, IntrinsicGuided)
                elif corr_type == 'neg':
                    self.S2_neg.addAntiTraces(Trace, IntrinsicGuided)

    def addWeakTrace(self, Trace, sensor, corr_type):
        if not self.established:
            # Guardo solo hasta donde se cumple la correlacion
            for i in reversed(range(len(Trace))):
                if corr_type == 'neg':
                    if Trace[i][sensor - 1] >= Trace[i - 1][sensor - 1]:
                        break
                elif corr_type == 'pos':
                    if Trace[i][sensor - 1] <= Trace[i - 1][sensor - 1]:
                        break

            if sensor == 1:
                if corr_type == 'pos':
                    self.S1_pos.addWeakTraces(Trace[i:])
                elif corr_type == 'neg':
                    self.S1_neg.addWeakTraces(Trace[i:])
            elif sensor == 2:
                if corr_type == 'pos':
                    self.S2_pos.addWeakTraces(Trace[i:])
                elif corr_type == 'neg':
                    self.S2_neg.addWeakTraces(Trace[i:])

    def isCorrelationEstablished(self):

        corr1_pos = self.S1_pos.numberOfGoalsWithoutAntiTraces
        corr1_neg = self.S1_neg.numberOfGoalsWithoutAntiTraces
        corr2_pos = self.S2_pos.numberOfGoalsWithoutAntiTraces
        corr2_neg = self.S2_neg.numberOfGoalsWithoutAntiTraces

        max_traces = max(corr1_pos, corr1_neg, corr2_pos, corr2_neg)
        if not self.established:
            if max_traces >= self.Tb_max:
                self.established = 1
                i = (corr1_pos, corr1_neg, corr2_pos, corr2_neg).index(max_traces)
                if i < 2:
                    self.corr_established = 1  # Sensor 1
                else:
                    self.corr_established = 2  # Sensor 2
                if i % 2 == 0:  # Posicion par
                    self.corr_established_type = 'pos'
                else:
                    self.corr_established_type = 'neg'
            else:
                self.established = 0
