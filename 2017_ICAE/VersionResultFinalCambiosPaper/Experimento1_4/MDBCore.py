from Simulador import *
from Episode import *
from CandidateStateEvaluator import *
from TracesBuffer import *
from CorrelationsManager import *
from StateSpace import *
import logging
import pickle
from matplotlib import colors


class MDBCore(object):
    def __init__(self):

        # Object initialization
        self.memoryVF = TracesBuffer()
        self.memoryVF.setMaxSize(50)  #100
        self.TracesMemoryVF = TracesMemory()
        # self.StateSpace = StateSpace()
        self.simulator = Sim()
        self.tracesBuffer = TracesBuffer()
        self.tracesBuffer.setMaxSize(50)  # 15
        self.intrinsicMemory = EpisodicBuffer()
        self.intrinsicMemory.setMaxSize(20)

        self.episode = Episode()
        self.correlationsManager = CorrelationsManager()
        self.CSE = CandidateStateEvaluator()

        self.stop = 0
        self.iterations = 0
        self.it_reward = 0  # Number of iteraration before obtaining reward
        self.it_blind = 0  # Number of iterations the Intrinsic blind motivation is active
        self.n_execution = 1  # Number of the execution of the experiment

        self.activeMot = 'Int'  # Variable to control the active motivation: Intrinsic ('Int') or Extrinsic ('Ext')

        self.activeCorr = 0  # Variable to control the active correlation. It contains its index
        self.corr_sensor = 0  # 1 - Sensor 1, 2 - Sensor 2, ... n- sensor n, 0 - no hay correlacion
        self.corr_type = ''  # 'pos' - Positive correlation, 'neg' - Negative correlation, '' - no correlation

        self.iter_min = 0  # Minimum number of iterations to consider possible an antitrace

        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, filename='LogFile.log')
        logging.info('Iteration  ActiveMotivation  ActiveCorrelation  CorrelatedSensor  CorrelationType  Episode')

        self.noMotivManager = 0

        # Graph matrixes
        self.graph1 = []
        self.graph2 = []
        # self.graph3 = []
        # self.graph4 = []
        # self.graph5 = []
        # self.graph6 = []
        # self.graph7 = []

        self.graphExec = []
        self.graphx = []

        # self.probUseGuided = 0  # Probability of using Intrinsic guided motivation during the Extrinsic use
        self.useVF = 0  # Indicate which Utility Model use (VF=1 or SUR=0) when Extrinsic motivation is active
        self.n_new_traces = 1001

    def run(self):

        # Save/load seed

        # Import seed
        f = open('seed_robobo3.pckl', 'rb')  # f = open('seed_robobo3.pckl', 'rb')
        seed = pickle.load(f)
        f.close()
        np.random.set_state(seed)

        # #Save seed
        # seed = np.random.get_state()
        # f = open('seed_robobo_exp.pckl', 'wb')
        # pickle.dump(seed, f)
        # f.close()

        self.main()

        # Save/close logs

    def main(self):

        self.stop = 0
        self.iterations = 0

        ########### WHEN LOADING DATA
        # self.loadData()
        # self.iterations = 51801#60001
        # action = self.CSE.actionChooser.getCandidateActions()[0]
        # self.activeMot = 'Ext' # Only when trying to use VF
        ###########

        while not self.stop:

            if self.iterations == 0:
                action = self.CSE.actionChooser.getCandidateActions(1)[0]

            # Sensorization in t (distances, action and motivation)
            self.episode.setSensorialStateT(self.simulator.get_sensorization())
            self.episode.setAction(action)
            # self.episode.setMotivation(self.activeMot)

            # self.simulator.baxter_larm_action(action)
            self.simulator.apply_action(action, self.iterations)
            # self.StateSpace.draw_state_space(self.simulator.get_sensorization())

            # Sensorization in t+1 (distances and reward)
            self.episode.setSensorialStateT1(self.simulator.get_sensorization())

            #####POSIBLE CAMBIO
            self.episode.setReward(self.simulator.get_reward())
            #####

            # Save episode in the pertinent memories
            self.tracesBuffer.addEpisode(self.episode.getEpisode())
            self.intrinsicMemory.addEpisode(self.episode.getSensorialStateT1())

            self.memoryVF.addEpisode(self.episode.getEpisode())


            ###########################
            if self.iterations > 0:
                self.writeLogs()
                if self.iterations % 1000 == 0:
                    self.debugPrint()
                self.saveGraphs()
            ###########################

            # Miro en el goal manager si hay reward y luego en la gestion de memoria elijo en
            # donde se deben guardar los datos corespondientes a eso

            # Check if a new correlation is needed or established
            self.correlationsManager.newCorrelation()
            if self.correlationsManager.correlations[self.activeCorr].i_reward_assigned == 0:
                self.correlationsManager.assignRewardAssigner(self.activeCorr, self.episode.getSensorialStateT1())

            # for i in range(len(self.correlationsManager.correlations)):
            #     self.correlationsManager.correlations[i].isCorrelationEstablished()
            #
            # if len(self.correlationsManager.correlations) > 1:
            #     if self.correlationsManager.correlations[1].S3_neg.numberOfGoalsWithoutAntiTraces > 6:
            #         print "AAAAA"

            # Memory Manager (Traces, weak traces and antitraces)
            if self.activeMot == 'Int':
                self.it_blind += 1
                self.noMotivManager = 0
                # If there is a reward, realise reward assignment and save trace in Traces Memory
                if self.episode.getReward():
                    self.simulator.restart_scenario()  # Restart scenario

                    self.correlationsManager.correlations[self.activeCorr].correlationEvaluator(
                        self.tracesBuffer.getTrace())  # Ya guardo aqui la traza debil

                    # self.activeCorr = len(self.correlationsManager.correlations) - 1

                    # self.activeCorr = self.correlationsManager.getActiveCorrelationPrueba(self.episode.getSensorialStateT1())
                    self.activeCorr = self.correlationsManager.getActiveCorrelationPrueba(
                        self.simulator.get_sensorization())  # Prueba provisional, despues debo seguir pasandole el sensorialStateT1, por lo que lo debo actuallizar al reiniciar el escenario
                    self.reinitializeMemories()
                    logging.info('Goal reward when Intrinsic Motivation')
                    # logging.info('State used to calculate the new active motivation: %s',
                    #             self.episode.getSensorialStateT1())
                    # logging.info('Real state: %s', self.simulator.get_sensorization())

                    self.it_reward = 0
                    self.it_blind = 0
                    self.n_execution += 1

                    self.saveMatrix()

                    self.TracesMemoryVF.addTraces(self.memoryVF.getTraceReward())
                    self.memoryVF.removeAll()

                elif self.correlationsManager.getReward(self.activeCorr, self.simulator.get_reward(),
                                                        tuple(self.episode.getSensorialStateT1())):
                    self.correlationsManager.correlations[self.activeCorr].correlationEvaluator(
                        self.tracesBuffer.getTrace())
                    # The active correlation is now the correlation that has provided the reward
                    self.activeCorr = self.correlationsManager.correlations[self.activeCorr].i_reward
                    self.reinitializeMemories()
                    logging.info('Correlation reward when Intrinsic Motivation')

            elif self.activeMot == 'Ext':
                self.noMotivManager = 1
                if self.episode.getReward():  # GOAL MANAGER - Encargado de asignar la recompensa?
                    self.simulator.restart_scenario()  # Restart scenario
                    # Save as trace in TracesMemory of the correlated sensor
                    self.correlationsManager.correlations[self.activeCorr].addTrace(self.tracesBuffer.getTrace(),
                                                                                    self.corr_sensor, self.corr_type)
                    # self.activeCorr = len(self.correlationsManager.correlations) - 1

                    # self.activeCorr = self.correlationsManager.getActiveCorrelationPrueba(self.episode.getSensorialStateT1())
                    self.activeCorr = self.correlationsManager.getActiveCorrelationPrueba(
                        self.simulator.get_sensorization())  # Prueba provisional, despues debo seguir pasandole el sensorialStateT1, por lo que lo debo actuallizar al reiniciar el escenario
                    # Quizas poido restar scenario e ao salir de esto da motivacion activa, antes do motiv manager, comprobar cal e a activeCorrelation, mirar se eso me interfire con algo do que faigo antes
                    self.reinitializeMemories()
                    logging.info('Goal reward when Extrinsic Motivation')
                    # logging.info('State used to calculate the new active motivation: %s',
                    #              self.episode.getSensorialStateT1())
                    # logging.info('Real state: %s', self.simulator.get_sensorization())

                    self.noMotivManager = 0
                    # self.activeMot = 'Int'

                    self.it_reward = 0
                    self.it_blind = 0
                    self.n_execution += 1

                    self.saveMatrix()

                    self.TracesMemoryVF.addTraces(self.memoryVF.getTraceReward())
                    self.memoryVF.removeAll()

                elif self.correlationsManager.getReward(self.activeCorr, self.simulator.get_reward(),
                                                        tuple(self.episode.getSensorialStateT1())):
                    # Save as trace in TracesMemory of the correlated sensor
                    self.correlationsManager.correlations[self.activeCorr].addTrace(self.tracesBuffer.getTrace(),
                                                                                    self.corr_sensor, self.corr_type)
                    # The active correlation is now the correlation that has provided the reward
                    self.activeCorr = self.correlationsManager.correlations[self.activeCorr].i_reward
                    self.reinitializeMemories()
                    logging.info('Correlation reward when Extrinsic Motivation')

                    self.noMotivManager = 0
                else:
                    # Check if the the active correlation is still active
                    if self.iter_min > 2:
                        sens_t = self.tracesBuffer.getTrace()[-2][self.corr_sensor - 1]
                        sens_t1 = self.tracesBuffer.getTrace()[-1][self.corr_sensor - 1]
                        dif = sens_t1 - sens_t

                        if (self.corr_type == 'pos' and dif <= 0) or (self.corr_type == 'neg' and dif >= 0):
                            # Guardo antitraza en el sensor correspondiente y vuelvo a comezar el bucle
                            self.correlationsManager.correlations[self.activeCorr].addAntiTrace(
                                self.tracesBuffer.getTrace(), self.corr_sensor, self.corr_type)

                            # self.activeCorr = len(self.correlationsManager.correlations) - 1

                            # self.activeCorr = self.correlationsManager.getActiveCorrelationPrueba(self.episode.getSensorialStateT1())
                            self.activeCorr = self.correlationsManager.getActiveCorrelationPrueba(
                                self.simulator.get_sensorization())  # Prueba provisional, despues debo seguir pasandole el sensorialStateT1, por lo que lo debo actuallizar al reiniciar el escenario
                            # self.tracesBuffer.removeAll()  # Reinitialize traces buffer
                            # self.iter_min = 0
                            self.reinitializeMemories()
                            logging.info('Antitrace in sensor %s of type %s', self.corr_sensor, self.corr_type)
                            logging.info('Sens_t %s, sens_t1 %s, diff %s', sens_t, sens_t1, dif)

                            # logging.info('State used to calculate the new active motivation: %s',
                            #              self.episode.getSensorialStateT1())
                            # logging.info('Real state: %s', self.simulator.get_sensorization())

                            self.noMotivManager = 0

            ### Motiv. Manager
            ### | | | |
            ### v v v v
            self.MotivationManager()
            ### ^ ^ ^ ^
            ### | | | |
            ### Motiv. Manager

            # CANDIDATE STATE EVALUATOR and ACTION CHOOSER
            # Generate new action
            SimData = (
                self.simulator.baxter_larm_get_pos(), self.simulator.baxter_larm_get_angle(),
                self.simulator.robobo_get_pos(), self.simulator.robobo_get_angle(),
                self.simulator.ball_get_pos(),
                self.simulator.ball_position, self.simulator.box1_get_pos())


            if self.iterations > 600000:
                self.useVF = 1

            # if self.episode.getReward() and self.n_new_traces > 1000:
            if self.n_new_traces > 10:
                trainNet=1
                self.n_new_traces = 0
            else:
                trainNet=0
                if self.episode.getReward():
                    self.n_new_traces += 1

            # action = self.CSE.getAction(self.activeMot, SimData, tuple(self.episode.getSensorialStateT1()),
            #                             self.corr_sensor, self.corr_type, self.intrinsicMemory.getContents())
            action = self.CSE.getAction(self.activeMot, SimData, tuple(self.simulator.get_sensorization()),
                                        self.corr_sensor, self.corr_type,
                                        self.intrinsicMemory.getContents(), self.useVF, self.TracesMemoryVF.getTracesList(), trainNet)  # Prueba mientras no solucione lo del reward de scenario y actualizar el estado s(t+1)
            # Others
            # self.writeLogs()
            # self.debugPrint()
            self.iter_min += 1
            self.iterations += 1
            self.it_reward += 1
            self.stopCondition()
            self.episode.cleanEpisode()

        self.saveData()

    def stopCondition(self):

        if self.iterations > 20000:  #60000:#62000:#25000:#60000:#90000:
            self.stop = 1

    def writeLogs(self):
        logging.debug('%s  -  %s  -  %s  -  %s  -  %s  -  %s', self.iterations, self.activeMot, self.activeCorr,
                      self.corr_sensor, self.corr_type, self.episode.getEpisode())

    def debugPrint(self):
        # print '------------------'
        print "Iteration: ", self.iterations
        # print "Active correlation: ", self.activeCorr
        # print "Active motivation: ", self.activeMot
        # print "Correlated sensor: ", self.corr_sensor, self.corr_type
        # print "Trazas consecutivas S3 neg: ", self.correlationsManager.correlations[
        #     self.activeCorr].S3_neg.numberOfGoalsWithoutAntiTraces
        # print "Trazas consecutivas S2 neg: ", self.correlationsManager.correlations[
        #     self.activeCorr].S2_neg.numberOfGoalsWithoutAntiTraces
        # print "Trazas consecutivas S1 neg: ", self.correlationsManager.correlations[
        #     self.activeCorr].S1_neg.numberOfGoalsWithoutAntiTraces

    def reinitializeMemories(self):
        self.tracesBuffer.removeAll()  # Reinitialize traces buffer
        self.iter_min = 0
        self.intrinsicMemory.removeAll()  # Reinitialize intrinsic memory
        self.intrinsicMemory.addEpisode(self.episode.getSensorialStateT1())

    def MotivationManager(self):
        self.activeMot = 'Int'
        # # self.activeMot = 'Ext'
        # if not self.noMotivManager:
        #     # self.corr_sensor, self.corr_type = self.correlationsManager.getActiveCorrelation(
        #     #     tuple(self.episode.getSensorialStateT1()), self.activeCorr)
        #     self.corr_sensor, self.corr_type = self.correlationsManager.getActiveCorrelation(
        #         tuple(self.simulator.get_sensorization()),
        #         self.activeCorr)  # Prueba, mientras no actualizo el tema del estado despues del reward de scenario
        #     if self.corr_sensor == 0:
        #         self.activeMot = 'Int'
        #     else:
        #         if self.activeMot == 'Int':
        #             # self.tracesBuffer.removeAll()
        #             self.iter_min = 0
        #         self.activeMot = 'Ext'

    def saveGraphs(self):
        # Graph 1 - Iterations to reach the goal vs. Total number of iterations
        self.graph1.append(
            (self.iterations, self.episode.getReward(), self.it_reward, self.it_blind,
             len(self.correlationsManager.correlations), self.activeMot, self.activeCorr,
             self.episode.getSensorialStateT1()))

        self.graph2.append((self.correlationsManager.correlations[-1].S1_neg.numberOfGoalsWithoutAntiTraces,
                            self.correlationsManager.correlations[-1].S1_pos.numberOfGoalsWithoutAntiTraces,
                            self.correlationsManager.correlations[-1].S2_neg.numberOfGoalsWithoutAntiTraces,
                            self.correlationsManager.correlations[-1].S2_pos.numberOfGoalsWithoutAntiTraces,
                            self.correlationsManager.correlations[-1].S3_neg.numberOfGoalsWithoutAntiTraces,
                            self.correlationsManager.correlations[-1].S3_pos.numberOfGoalsWithoutAntiTraces))

        # Save executions
        self.graphExec.append(
            (self.iterations, self.episode.getReward(), self.it_reward, self.it_blind,
             len(self.correlationsManager.correlations), self.activeMot, self.activeCorr,
             self.episode.getSensorialStateT1()))

    def saveMatrix(self):
        self.graphx.append(self.graphExec)
        self.graphExec = []

    def plotGraphs(self):
        # Graph 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n_reward = 0
        iter_goal = []  # Number of iteration increase at the same time as goals
        for i in range(len(self.graph1)):
            # for i in range(40000):
            if self.graph1[i][1]:
                n_reward += 1
                iter_goal.append(self.graph1[i][0])
                if self.graph1[i][7][1] == 0.0:  # Distance Baxter-ball=0.0 when reward
                    # plt.plot(self.graph1[i][0], self.graph1[i][2], 'ro', color='red')
                    ax.plot(n_reward, self.graph1[i][2], 'ro', color='red')
                else:  # The reward is given to the Robobo
                    # plt.plot(self.graph1[i][0], self.graph1[i][2], 'ro', color='blue')
                    # ax.plot(n_reward, self.graph1[i][2], 'ro', color='blue')
                    ax.plot(n_reward, self.graph1[i][2], 'x', color='black', markeredgewidth=3.5, markersize=12.0)
            if self.graph1[i][4] > self.graph1[i - 1][4]:
                ax.axvline(x=n_reward)

        ax.axvline(x=347, linewidth=2.0, color='green', linestyle='--')
        ax.axvline(x=363+5, linewidth=2.0, color='purple', linestyle='--')
        ax.axvline(x=379.5, linewidth=2.5, color='grey', linestyle='--')
        # ax.axvline(x=220.5, linewidth=2.0, color='grey', linestyle='--')# ax.axvline(x=378, linewidth=2.0, color='orange', linestyle='--')
        # ax.axvline(x=506.5, linewidth=2.0, color='purple', linestyle='--')
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         plt.axvline(x=self.graph1[i][0])

        # plt.axes().set_xlabel('Iterations')
        ax.set_xlabel('Rewards', size=15.0)
        ax.set_ylabel('Iterations needed to reach the goal', size=17.0)
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(range(n_reward), iter_goal, marker='.', markersize=1.0, color='green', linewidth=1.0, label='active')
        ax2.set_ylabel('Number of iterations', size=15.0)
        # plt.xlim(0, 530+5)
        # plt.xlim(0, 55000)
        plt.show()
        #Simple moving average
        reward_matrix = []
        blind_matrix = []
        blind_matrix2 = []
        blind_matrix3 = []
        iter = []
        iter2 = []
        iter3 = []
        iter4 = []
        for i in range(len(self.graph1)):
            # for i in range(40000):
            if self.graph1[i][1]:
                reward_matrix.append(self.graph1[i][2])
                # blind_matrix.append(self.graph1[i][3])
                # if self.graph1[i][3] == 0: # 1 = use of Ib, 0 = no use of Ib
                #     blind_matrix.append(0)
                # else:
                #     blind_matrix.append(1*100)
                if self.graph1[i][0] < 50001:
                    blind_matrix.append(min(self.graph1[i][3] / 100.0, 1))
                    iter2.append(self.graph1[i][0])
                if self.graph1[i][0] > 50000 and self.graph1[i][0] < 50770:
                    blind_matrix2.append(min(self.graph1[i][3] / 100.0, 1))
                    iter3.append(self.graph1[i][0])
                if self.graph1[i][0] > 50770:
                    blind_matrix3.append(min(self.graph1[i][3] / 100.0, 1))
                    iter4.append(self.graph1[i][0])
                iter.append(self.graph1[i][0])
        window = 100
        window_aux = 50
        media = self.calcSma(reward_matrix, window)
        media_aux = self.calcSma(reward_matrix[:window - 1], window_aux)
        media_sum = media_aux + media[window - 1:]
        # plt.plot(iter, media_sum, marker='.', color='cyan', linewidth=0.5, label='simple moving average')
        ax.plot(range(n_reward), media_sum, marker='.', markersize=2.0, color='cyan', linewidth=2.0, label='simple moving average')
        print media_sum[50], media_sum[51], media_sum[49]
        print media_sum[-1]
        print media_sum[205], media_sum[204], media_sum[206]

        # # Accumulated reward
        # acum = []
        # aux = 0
        # for i in range(len(self.graph1)):
        #     aux += self.graph1[i][1]
        #     acum.append(aux)
        # plt.figure()
        # plt.plot(range(len(acum)), acum, marker='.', color=(0.0, 0.0, 0.6), linewidth=3.0, label='accumulated reward')
        #
        # # Simple moving average accumulated reward
        # window = 1500
        # window_aux = 50
        # media = self.calcSma(acum, window)
        # media_aux = self.calcSma(acum[:window - 1], window_aux)
        # media_sum = media_aux + media[window - 1:]
        # plt.plot(range(len(media_sum)), media_sum, marker='.', color='cyan', linewidth=0.5, label='sma acumulated reward')
        # # print len(media_sum), len(acum)
        # plt.legend()
        # # Rewards/iterations
        # for i in range(len(media_sum)):
        #     if media_sum[i]==None:
        #         media_sum[i]=0.0
        # ventana = 500
        # inc = []
        # for i in range(len(acum) / ventana - 1):
        #     inc.append(acum[i * ventana + ventana] - acum[i * ventana])
        # inc = [0]+inc
        # plt.figure()
        # plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 0.0, 0.6), linewidth=0.5, label='inc accumulated reward')
        # ###
        # inc = []
        # media_sum=list(media_sum)
        # for i in range(len(media_sum) / ventana - 1):
        #     inc.append(media_sum[i * ventana + ventana] - media_sum[i * ventana])
        # inc = [0]+inc
        # plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 1.0, 0.6), linewidth=0.5, label='sma inc accumulated reward')
        # plt.legend()
        # #########
        # plt.figure()
        # acum_rew = np.array(range(len(iter) + 1))
        # it = np.array([0] + iter)
        # plt.plot([0] + iter, list(it/acum_rew), marker='.', color=(0.6, 0.0, 0.6), linewidth=0.5, label='iterations/accumulated reward')
        # plt.legend()
        # plt.figure()
        # plt.plot([0] + iter, list(acum_rew.astype(float) / it.astype(float)), marker='.', color=(0.0, 0.6, 0.6), linewidth=0.5,
        #          label='accumulated reward/iterations')
        # plt.legend()


        # Graph 2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.graph1)):
            if self.graph1[i][1]:
                ax.plot(self.graph1[i][0], self.graph1[i][3], 'ro', color='green')
        for i in range(len(self.graph1)):
            if self.graph1[i][4] > self.graph1[i - 1][4]:
                ax.axvline(x=self.graph1[i][0])
        ax.set_xlabel('Iterations', size=15.0)
        ax.set_ylabel('Iterations the Ib motivation is active', size=15.0)
        ax.grid()
        ax.axvline(x=50000, linewidth=2.0, color='green', linestyle='--')
        ax.axvline(x=50483+287, linewidth=2.0, color='purple', linestyle='--')
        ax.axvline(x=51800, linewidth=2.5, color='grey', linestyle='--')
        # ax.axvline(x=40000, linewidth=2.0, color='grey', linestyle='--')
        # ax.axvline(x=60000, linewidth=2.0, color='purple', linestyle='--')
        ax2 = ax.twinx()
        # Simple moving average: 1 = use of Ib, 0 = no use of Ib
        window = 25
        window_aux = 4
        media = self.calcSmac(blind_matrix, window)
        # media_aux = self.calcSmac(blind_matrix[:window - 1], window_aux)
        # media_sum = media_aux + media[window - 1:]
        print 'Vector', blind_matrix
        # ax2.plot(iter, media_sum, marker='.', markersize=1.0, color='orange', linewidth=1.0, label='active')
        ax2.plot(iter2, media, marker='.', markersize=1.0, color='orange', linewidth=1.5, label='active')
        print 'media', media
        print blind_matrix3

        window = 10
        window_aux = 4
        media2 = self.calcSmac(blind_matrix2 +blind_matrix3, window)
        # media_aux2 = self.calcSmac(blind_matrix2[:window - 1], window_aux)
        # media_sum2 = media_aux2 + media2[window - 1:]

        ax2.plot(iter3+iter4, media2, marker='.', markersize=1.0, color='orange', linewidth=1.5, label='active')

        # window = 5
        # window_aux = 4
        # media3 = self.calcSmac(blind_matrix3, window)
        # # media_aux3 = self.calcSmac(blind_matrix3[:window - 1], window_aux)
        # # media_sum3 = media_aux3 + media3[window - 1:]
        #
        # ax2.plot(iter4, media3, marker='.', markersize=1.0, color='orange', linewidth=2.0, label='active')

        ax2.set_ylabel('Use of Ib', size=15.0)
        plt.xlim(0, 62000)#plt.xlim(0, 53000)
        plt.show()

        # Graph 3
        plt.figure()
        contS1neg = []
        contS1pos = []
        contS2neg = []
        contS2pos = []
        contS3neg = []
        contS3pos = []
        for i in range(len(self.graph2)):
            contS1neg.append(self.graph2[i][0])
            contS1pos.append(self.graph2[i][1])
            contS2neg.append(self.graph2[i][2])
            contS2pos.append(self.graph2[i][3])
            contS3neg.append(self.graph2[i][4])
            contS3pos.append(self.graph2[i][5])
            # if self.graph1[i][4] > self.graph1[i - 1][4]:
            if max(self.graph2[i]) > 19:
                plt.axvline(x=i + 1, linewidth=2.0)
        plt.plot(range(len(self.graph2)), contS1neg, marker='.', markersize=0.5, linewidth=0.5, color='cyan',
                 label='dCR-')
        plt.plot(range(len(self.graph2)), contS1pos, marker='.', markersize=0.5, linewidth=0.5, color='brown',
                 label='dCR+')
        plt.plot(range(len(self.graph2)), contS2neg, marker='.', markersize=0.5, linewidth=0.5, color='green',
                 label='dCB-')
        plt.plot(range(len(self.graph2)), contS2pos, marker='.', markersize=0.5, linewidth=0.5, color='purple',
                 label='dCB+')
        plt.plot(range(len(self.graph2)), contS3neg, marker='.', markersize=0.5, linewidth=0.5, color='red',
                 label='dCX-')
        plt.plot(range(len(self.graph2)), contS3pos, marker='.', markersize=0.5, linewidth=0.5, color='orange',
                 label='dCX+')
        plt.axes().set_xlabel('Iterations', size=15.0)
        plt.axes().set_ylabel('Balance between Successful and Failed Traces', size=15.0)
        plt.ylim(0, 25)
        plt.xlim(0, 62000)
        # plt.xlim(0, 40000)
        plt.grid()
        plt.legend()

    def plotScaffoldingMap(self):
        f = open('scaffoldingMatrixReal.pckl', 'rb')
        ScaffoldingMatrix = pickle.load(f)
        f.close()
        print ScaffoldingMatrix
        # Correlation 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j][k] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j][k] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j][k] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j][k] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'orange' and color != 'blue' and color != 'cyan':
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('dCR (m)', size=18.0)
        ax.set_ylabel('dCB (m)', size=18.0)
        ax.set_zlabel('dCX (m)', size=18.0)
        ax.xaxis.set_ticks([0, 4, 8, 12])
        ax.set_xticklabels(['0', '0.4', '0.8', '1.2'])
        ax.yaxis.set_ticks([0, 4, 8, 12])
        ax.set_yticklabels(['0', '0.4', '0.8', '1.2'])
        ax.zaxis.set_ticks([0, 4, 8])
        ax.set_zticklabels(['0', '0.4', '0.8'])
        ax.tick_params(labelsize=18.0)
        plt.show()
        # Correlation 1+2
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j][k] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j][k] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j][k] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j][k] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'blue' and color != 'cyan':
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('dCR (m)', size=18.0)
        ax.set_ylabel('dCB (m)', size=18.0)
        ax.set_zlabel('dCX (m)', size=18.0)
        ax.xaxis.set_ticks([0, 4, 8, 12])
        ax.set_xticklabels(['0', '0.4', '0.8', '1.2'])
        ax.yaxis.set_ticks([0, 4, 8, 12])
        ax.set_yticklabels(['0', '0.4', '0.8', '1.2'])
        ax.zaxis.set_ticks([0, 4, 8, 12])
        ax.set_zticklabels(['0', '0.4', '0.8', '1.2'])
        ax.tick_params(labelsize=18.0)
        plt.show()
        # Correlation 1+2+3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j][k] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j][k] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j][k] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j][k] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'cyan':
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('dCR (m)', size=18.0)
        ax.set_ylabel('dCB (m)', size=18.0)
        ax.set_zlabel('dCX (m)', size=18.0)
        ax.xaxis.set_ticks([0, 4, 8, 12])
        ax.set_xticklabels(['0', '0.4', '0.8', '1.2'])
        ax.yaxis.set_ticks([0, 4, 8, 12])
        ax.set_yticklabels(['0', '0.4', '0.8', '1.2'])
        ax.zaxis.set_ticks([0, 4, 8, 12])
        ax.set_zticklabels(['0', '0.4', '0.8', '1.2'])
        ax.tick_params(labelsize=18.0)
        plt.show()
        # Correlation 1+2+3+4
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j][k] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j][k] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j][k] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j][k] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red':
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        # ax.set_xlabel('dCR ($10^{-1}$ m)', size=15.0)
        # ax.set_ylabel('dCB ($10^{-1}$ m)', size=15.0)
        # ax.set_zlabel('dCX ($10^{-1}$ m)', size=15.0)
        ax.set_xlabel('dCR (m)', size=18.0)
        ax.set_ylabel('dCB (m)', size=18.0)
        ax.set_zlabel('dCX (m)', size=18.0)
        ax.xaxis.set_ticks([0, 4, 8, 12])
        ax.set_xticklabels(['0', '0.4', '0.8', '1.2'])
        ax.yaxis.set_ticks([0, 4, 8, 12])
        ax.set_yticklabels(['0', '0.4', '0.8', '1.2'])
        ax.zaxis.set_ticks([0, 4, 8, 12])
        ax.set_zticklabels(['0', '0.4', '0.8', '1.2'])
        ax.tick_params(labelsize=18.0)
        plt.show()
        # Correlation 1+2+3+4 same colour
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j][k] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j][k] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j][k] == 3.0 or ScaffoldingMatrix[i][j][k] == 4.0:
                        color = 'blue'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red':
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('dCR (m)', size=18.0)
        ax.set_ylabel('dCB (m)', size=18.0)
        ax.set_zlabel('dCX (m)', size=18.0)
        ax.xaxis.set_ticks([0, 4, 8, 12])
        ax.set_xticklabels(['0', '0.4', '0.8', '1.2'])
        ax.yaxis.set_ticks([0, 4, 8, 12])
        ax.set_yticklabels(['0', '0.4', '0.8', '1.2'])
        ax.zaxis.set_ticks([0, 4, 8, 12])
        ax.set_zticklabels(['0', '0.4', '0.8', '1.2'])
        ax.tick_params(labelsize=18.0)
        plt.show()

    def calcSma(sel, data, smaPeriod):
        j = next(i for i, x in enumerate(data) if x is not None)
        our_range = range(len(data))[j + smaPeriod - 1:]
        empty_list = [None] * (j + smaPeriod - 1)
        sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

        return list(empty_list + sub_result)

    def calcSmac(self, vector, window):
        """Media movil centrada y ajustable de un vector y con ventana 'window' hacia ambos lados"""
        smac=[]
        for i in range(len(vector)):
            Linf = i-window
            Lsup = i+window+1
            window_sup = window
            if Linf < 0:
                Linf = 0
                Lsup += (window-i)
            elif Lsup > len(vector)-1:
                Linf -= (Lsup-len(vector))
                Lsup = len(vector)
                # window_sup=len(vector)-i-1
            smac.append(sum(vector[Linf:Lsup])/float(window+window_sup+1))
        return smac

    def plotEpisode(self, graph, name):
        sens1 = []
        sens2 = []
        sens3 = []
        active_corr = []
        for i in range(len(graph)):
            # Reorganize sensor values to plot them
            sens1.append(graph[i][7][0])
            sens2.append(graph[i][7][1])
            sens3.append(graph[i][7][2])
            if graph[i][5] == 'Int':
                active_corr.append(100)
            else:
                active_corr.append((graph[-1][4] - graph[i][6] + 1) * 100)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.title('Representative execution: ' + str(name))
        cn = colors.Normalize(min(active_corr), max(active_corr))  # creates a Normalize object for these z values
        for i in range(len(sens1)):
            fi = 5.0 - (len(sens1) - i) / len(sens1)
            ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5,
                    marker='.', markersize=fi)
            plt.show()

    def plotEpisodes(self, graph, Lsup, Linf):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('dCR (m)')
        ax.set_ylabel('dCB (m)')
        ax.set_zlabel('dCX (m)')
        # plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                # active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    # if graph[i][j][5] == 'Int':
                    #     active_corr.append(100)
                    # else:
                    #     active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                    # cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                    # for k in range(len(sens1)):
                    #     fi = 5.0 - (len(sens1) - k) / len(sens1)
                    #     coef_deg = 1.0 - min((len(sens1) - k) / 250.0, 1.0)
                    #     if k == len(sens1) - 1:
                    #         color = 'red'
                    #         markers = 6.0
                    #     elif k == 1:
                    #         color = 'green'
                    #         markers = 6.0
                    #     else:
                    #         color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                    #         markers = fi
                    #     # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    #     ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #             sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #             sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=color, linewidth=0.5,
                    #             marker='.', markersize=markers)
                    #     # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)
                    # #

                # cn = colors.Normalize(min(active_corr), max(active_corr))  # creates a Normalize object for these z values
                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0,min((len(sens1)-k)/100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)# color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                            sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                            sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg), linewidth=0.0,
                            marker='.', markersize=markers)
                    # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    plt.show()
        ax.xaxis.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
        ax.yaxis.set_ticks([0, 500, 1000, 1500, 2000, 2500])
        ax.set_yticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5'])
        ax.zaxis.set_ticks([0, 500, 1000, 1500, 2000])
        ax.set_zticklabels(['0', '0.5', '1.0', '1.5', '2.0'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = fig.gca(projection='3d')
        ax.set_xlabel('dCB (m)')
        ax.set_ylabel('dCX (m)')
        # ax.set_zlabel('Distance Basket - Cylinder')
        # plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    if graph[i][j][5] == 'Int':
                        active_corr.append(100)
                    else:
                        active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0, min((len(sens1) - k) / 100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         linewidth=0.0,
                    #         marker='.', markersize=markers)
                    # # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='y', zs=2.5)

                    # ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='x', zs=-2.5)
                    ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                            sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1],
                            color=plt.cm.jet(coef_deg), linewidth=0.0, marker='.', markersize=markers)
                    #
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='z', zs=-2.5)

                    plt.show()
        ax.xaxis.set_ticks([0, 500, 1000, 1500, 2000, 2500])
        ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5'])
        ax.yaxis.set_ticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
        plt.grid()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = fig.gca(projection='3d')
        ax.set_xlabel('dCB (m)')
        ax.set_ylabel('dCX (m)')
        # ax.set_zlabel('Distance Basket - Cylinder')
        # plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    if graph[i][j][5] == 'Int':
                        active_corr.append(100)
                    else:
                        active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0, min((len(sens1) - k) / 100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         linewidth=0.0,
                    #         marker='.', markersize=markers)
                    # # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='y', zs=2.5)

                    # ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='x', zs=-2.5)
                    if sens1[k] < 50:
                        ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                                sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1],
                                color=plt.cm.jet(coef_deg), linewidth=0.0, marker='.', markersize=markers)
                    #
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='z', zs=-2.5)

                    plt.show()
            plt.grid()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = fig.gca(projection='3d')
        ax.set_xlabel('dCR (m)')
        ax.set_ylabel('dCB (m)')
        # ax.set_zlabel('Distance Basket - Cylinder')
        # plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    if graph[i][j][5] == 'Int':
                        active_corr.append(100)
                    else:
                        active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0, min((len(sens1) - k) / 100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         linewidth=0.0,
                    #         marker='.', markersize=markers)
                    # # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='y', zs=2.5)

                    # ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='x', zs=-2.5)

                    ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                            sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1],
                            color=plt.cm.jet(coef_deg), linewidth=0.0, marker='.', markersize=markers)
                    #
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='z', zs=-2.5)

                    plt.show()
        ax.xaxis.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
        ax.yaxis.set_ticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
        plt.grid()


        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = fig.gca(projection='3d')
        ax.set_xlabel('dCR (m)')
        ax.set_ylabel('dCX (m)')
        # ax.set_zlabel('Distance Basket - Cylinder')
        # plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    if graph[i][j][5] == 'Int':
                        active_corr.append(100)
                    else:
                        active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0, min((len(sens1) - k) / 100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         linewidth=0.0,
                    #         marker='.', markersize=markers)
                    # # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='y', zs=2.5)

                    # ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='x', zs=-2.5)

                    if sens2[k] < 50:
                        ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                                sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                                linewidth=0.0, marker='.', markersize=markers)
                    #
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='z', zs=-2.5)

                    plt.show()
        ax.xaxis.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
        ax.yaxis.set_ticks([0, 200, 400, 600, 800])
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
        plt.grid()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = fig.gca(projection='3d')
        ax.set_xlabel('dCR (m)')
        ax.set_ylabel('dCB (m)')
        # ax.set_zlabel('Distance Basket - Cylinder')
        # plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    if graph[i][j][5] == 'Int':
                        active_corr.append(100)
                    else:
                        active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0, min((len(sens1) - k) / 100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         linewidth=0.0,
                    #         marker='.', markersize=markers)
                    # # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='y', zs=2.5)

                    # ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='x', zs=-2.5)

                    if sens2[k] > 400:
                        ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                                sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                                linewidth=0.0, marker='.', markersize=markers)
                    #
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='z', zs=-2.5)

                    plt.show()
        plt.grid()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = fig.gca(projection='3d')
        ax.set_xlabel('Distance Robobo - Cylinder')
        ax.set_ylabel('Distance Baxter - Cylinder')
        # ax.set_zlabel('Distance Basket - Cylinder')
        plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
        for i in range(len(graph)):
            if i <= Lsup and i > Linf:
                # color = ((i-Linf)/(Lsup-Linf), 0.5, 0.5)
                sens1 = []
                sens2 = []
                sens3 = []
                active_corr = []
                for j in range(len(graph[i])):
                    # Reorganize sensor values to plot them
                    sens1.append(graph[i][j][7][0])
                    sens2.append(graph[i][j][7][1])
                    sens3.append(graph[i][j][7][2])
                    if graph[i][j][5] == 'Int':
                        active_corr.append(100)
                    else:
                        active_corr.append((graph[i][-1][4] - graph[i][j][6] + 1) * 100)

                cn = colors.Normalize(0, len(graph[i]))  # creates a Normalize object for these z values
                for k in range(len(sens1)):
                    # fi = 5.0 - (len(sens1) - k) / len(sens1)
                    coef_deg = 1.0 - max(0, min((len(sens1) - k) / 100.0, 1.0))
                    coef_deg = math.pow(coef_deg, 1)
                    if coef_deg < 0.001:
                        color = 'red'
                        markers = 0.0
                    elif coef_deg > 0.999:
                        color = 'green'
                        markers = 0.0
                    else:
                        color = (1.0 - coef_deg, 1.0 - coef_deg, 0.6)  # color = (0.0, 0.0, 1.0-coef_deg)
                        markers = 10
                    # # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         linewidth=0.0,
                    #         marker='.', markersize=markers)
                    # # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='y', zs=2.5)

                    # ax.plot(sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                    #         sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='x', zs=-2.5)

                    ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                            sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                            color=plt.cm.jet(coef_deg), linewidth=0.0, marker='.', markersize=markers)
                    #
                    # ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                    #         sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                    #         zdir='z', zs=-2.5)

                    plt.show()

    def saveData(self):  # , action, seed):
        f = open('TracesNovelty2.pckl', 'wb')
        pickle.dump(len(self.correlationsManager.correlations), f)
        for i in range(len(self.correlationsManager.correlations)):
            pickle.dump(self.correlationsManager.correlations[i].S1_pos, f)
            pickle.dump(self.correlationsManager.correlations[i].S1_neg, f)
            pickle.dump(self.correlationsManager.correlations[i].S2_pos, f)
            pickle.dump(self.correlationsManager.correlations[i].S2_neg, f)
            pickle.dump(self.correlationsManager.correlations[i].S3_pos, f)
            pickle.dump(self.correlationsManager.correlations[i].S3_neg, f)
            pickle.dump(self.correlationsManager.correlations[i].corr_active, f)
            pickle.dump(self.correlationsManager.correlations[i].corr_type, f)
            pickle.dump(self.correlationsManager.correlations[i].established, f)
            pickle.dump(self.correlationsManager.correlations[i].corr_established, f)
            pickle.dump(self.correlationsManager.correlations[i].corr_established_type, f)
            pickle.dump(self.correlationsManager.correlations[i].i_reward, f)
            pickle.dump(self.correlationsManager.correlations[i].i_reward_assigned, f)
        # pickle.dump(self.activeMot, f)
        # pickle.dump(self.activeCorr, f)
        # pickle.dump(self.corr_sensor, f)
        # pickle.dump(self.corr_type, f)
        pickle.dump(self.graph1, f)
        pickle.dump(self.graphx, f)
        pickle.dump(self.graph2, f)
        # pickle.dump(self.graph3, f)
        # pickle.dump(self.graph4, f)
        # pickle.dump(self.graph5, f)
        # pickle.dump(self.graph6, f)
        # pickle.dump(self.graph7, f)
        pickle.dump(self.TracesMemoryVF, f)
        f.close()

    def loadData(self):
        f = open('SimulationDataRealFinal379repairedNorm.pckl', 'rb')  # SimulationDataLongExecBis2 SimulationDataRealFinal379repairedNorm
                                                            # SimulationDataLongExecBis2_changeGoalLong 90000
                                                            # SimulationDataLongExec_changeGoalLong 60000
                                                            # TracesNovelty 25000
                                                            # TracesVF 65000 (realmente 62000)
                                                            # TracesVF2 62000
                                                            # SimulationDataRealFinal379repairedNormChangeGoal 60000
                                                            # TracesNovelty2
        numero = pickle.load(f)
        for i in range(numero):
            self.correlationsManager.correlations.append(Correlations(None))
        for i in range(numero):
            self.correlationsManager.correlations[i].S1_pos = pickle.load(f)
            self.correlationsManager.correlations[i].S1_neg = pickle.load(f)
            self.correlationsManager.correlations[i].S2_pos = pickle.load(f)
            self.correlationsManager.correlations[i].S2_neg = pickle.load(f)
            self.correlationsManager.correlations[i].S3_pos = pickle.load(f)
            self.correlationsManager.correlations[i].S3_neg = pickle.load(f)
            self.correlationsManager.correlations[i].corr_active = pickle.load(f)
            self.correlationsManager.correlations[i].corr_type = pickle.load(f)
            self.correlationsManager.correlations[i].established = pickle.load(f)
            self.correlationsManager.correlations[i].corr_established = pickle.load(f)
            self.correlationsManager.correlations[i].corr_established_type = pickle.load(f)
            self.correlationsManager.correlations[i].i_reward = pickle.load(f)
            self.correlationsManager.correlations[i].i_reward_assigned = pickle.load(f)
        # self.activeMot = pickle.load(f)
        # self.activeCorr = pickle.load(f)
        # self.corr_sensor = pickle.load(f)
        # self.corr_type = pickle.load(f)
        self.graph1 = pickle.load(f)
        self.graphx = pickle.load(f)
        self.graph2 = pickle.load(f)
        # self.graph3 = pickle.load(f)
        # self.graph4 = pickle.load(f)
        # self.graph5 = pickle.load(f)
        # self.graph6 = pickle.load(f)
        # self.graph7 = pickle.load(f)
        # self.TracesMemoryVF = pickle.load(f)
        f.close()

    def saveScaffoldingMatrix(self):
        '''Ad hoc for the paper'''
        Map1 = self.correlationsManager.correlations[0].S3_neg.getCertaintyMatrix()
        Map2 = self.correlationsManager.correlations[1].S2_neg.getCertaintyMatrix()
        Map3 = self.correlationsManager.correlations[2].S1_neg.getCertaintyMatrix()
        # Map4 = self.correlationsManager.correlations[3].S1_neg.getCertaintyMatrix()
        # Map5 = self.correlationsManager.correlations[4].S2_neg.getCertaintyMatrix()

        # Create Scaffolding matrix
        ScaffoldingMatrix = Map1
        for x in range(len(Map2)):
            for y in range(len(Map2)):
                for z in range(len(Map2)):
                    if ScaffoldingMatrix[x][y][z] == 0.0:
                        if Map2[x][y][z]:
                            ScaffoldingMatrix[x][y][z] = 2
                        elif Map3[x][y][z]:
                            ScaffoldingMatrix[x][y][z] = 3
                        # elif Map4[x][y][z]:
                        #     ScaffoldingMatrix[x][y][z] = 4
                        # elif Map5[x][y][z]:
                        #     ScaffoldingMatrix[x][y][z] = 5
        print ScaffoldingMatrix

        f = open('scaffoldingMatrixReal.pckl', 'wb')
        pickle.dump(ScaffoldingMatrix, f)
        f.close()

    def DiffBetweenScaffMatrix(self):
        f = open('scaffoldingMatrixReal.pckl', 'rb')
        ScaffoldingMatrix = pickle.load(f)
        f.close()
        verdes_2 = 0
        naranjas_2 = 0
        azules_2 = 0
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 1.0:
                        verdes_2 += 1
                    elif ScaffoldingMatrix[i][j][k] == 2.0:
                        naranjas_2 += 1
                    elif ScaffoldingMatrix[i][j][k] == 3.0:
                        azules_2 += 1

        f = open('scaffoldingMatrixLongExec.pckl', 'rb')
        ScaffoldingMatrix_1 = pickle.load(f)
        f.close()
        verdes_1 = 0
        naranjas_1 = 0
        azules_1 = 0
        for i in range(len(ScaffoldingMatrix_1)):  # range(number):
            for j in range(len(ScaffoldingMatrix_1)):
                for k in range(len(ScaffoldingMatrix_1)):
                    if ScaffoldingMatrix_1[i][j][k] == 1.0:
                        verdes_1 += 1
                    elif ScaffoldingMatrix_1[i][j][k] == 2.0:
                        naranjas_1 += 1
                    elif ScaffoldingMatrix_1[i][j][k] == 3.0 or ScaffoldingMatrix_1[i][j][k] == 4.0:
                        azules_1 += 1

        # Bar plot
        N = 3
        men_means = (verdes_1, naranjas_1, azules_1)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, men_means, width, color='c')
        women_means = (verdes_2, naranjas_2, azules_2)
        rects2 = ax.bar(ind + width, women_means, width, color='y')
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Number of points in the certainty area')
        # ax.set_title('SUR')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('SUR1', 'SUR2', 'SUR3'))
        # ax.set_ylim(0,1300)
        ax.legend((rects1[0], rects2[0]), ('Simulation', 'After Sequential Real Learning'), loc='upper left')

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.show()

        # 3D representation - New zones
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                for k in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j][k] == 1.0 and ScaffoldingMatrix_1[i][j][k] != 1.0:
                        color = 'green'
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
                    elif ScaffoldingMatrix[i][j][k] == 2.0 and ScaffoldingMatrix_1[i][j][k] != 2.0:
                        color = 'orange'
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
                    elif ScaffoldingMatrix[i][j][k] == 3.0 and ScaffoldingMatrix_1[i][j][k] != 3.0 and ScaffoldingMatrix_1[i][j][k] != 4.0:
                        color = 'blue'
                        ax.scatter(i, j, k, c=color, marker='s', s=60,linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Distance Robobo-cylinder')
        ax.set_ylabel('Distance Baxter-cylinder')
        ax.set_zlabel('Distance basket-cylinder')
        ax.set_title('New zones')
        plt.show()

        # 3D representation - Deleted zones
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(ScaffoldingMatrix_1)):  # range(number):
            for j in range(len(ScaffoldingMatrix_1)):
                for k in range(len(ScaffoldingMatrix_1)):
                    if ScaffoldingMatrix_1[i][j][k] == 1.0 and ScaffoldingMatrix[i][j][k] != 1.0:
                        color = 'green'
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
                    elif ScaffoldingMatrix_1[i][j][k] == 2.0 and ScaffoldingMatrix[i][j][k] != 2.0:
                        color = 'orange'
                        ax.scatter(i, j, k, c=color, marker='s', s=60,
                                   linewidth=0.5)  # para mapa continuo sin s y sin linewidth
                    elif ScaffoldingMatrix_1[i][j][k] == 3.0 and ScaffoldingMatrix[i][j][k] != 3.0 and ScaffoldingMatrix[i][j][k] != 4.0:
                        color = 'blue'
                        ax.scatter(i, j, k, c=color, marker='s', s=60,linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Distance Robobo-cylinder')
        ax.set_ylabel('Distance Baxter-cylinder')
        ax.set_zlabel('Distance basket-cylinder')
        ax.set_title('Deleted zones')
        plt.show()

def main():
    instance = MDBCore()
    instance.run()


if __name__ == '__main__':
    main()


# import pickle;f=open('store.pckl', 'rb');seed=pickle.load(f);f.close();import numpy as np;np.random.set_state(seed);from MDBCore import *;a=MDBCore()
# a.run()
# seed=np.random.get_state();import pickle;f=open('seed_object.pckl', 'wb');pickle.dump(seed,f);f.close()
