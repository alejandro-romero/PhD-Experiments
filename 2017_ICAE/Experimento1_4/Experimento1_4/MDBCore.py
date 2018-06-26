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

        self.memoryVF = TracesBuffer()
        self.memoryVF.setMaxSize(100)
        self.TracesMemoryVF = TracesMemory()
        # Object initialization
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
        self.graphExec = []
        self.graphx = []

    def run(self):

        # Save/load seed

        # Import seed
        f = open('seed_robobo3.pckl', 'rb')
        seed = pickle.load(f)
        f.close()
        np.random.set_state(seed)

        #Save seed
        # seed = np.random.get_state()
        # f = open('seed_SimulationData11.pckl', 'wb')
        # pickle.dump(seed, f)
        # f.close()

        self.main()

        # Save/close logs

    def main(self):

        self.stop = 0
        self.iterations = 0

        while not self.stop:

            if self.iterations == 0:
                action = self.CSE.actionChooser.getCandidateActions(1)[0]

            # Sensorization in t (distances, action and motivation)
            self.episode.setSensorialStateT(self.simulator.get_sensorization())
            self.episode.setAction(action)
            # self.episode.setMotivation(self.activeMot)

            # self.simulator.baxter_larm_action(action)
            self.simulator.apply_action(action)
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
                # self.writeLogs()
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

            # action = self.CSE.getAction(self.activeMot, SimData, tuple(self.episode.getSensorialStateT1()),
            #                             self.corr_sensor, self.corr_type, self.intrinsicMemory.getContents())
            action = self.CSE.getAction(self.activeMot, SimData, tuple(self.simulator.get_sensorization()),
                                        self.corr_sensor, self.corr_type,
                                        self.intrinsicMemory.getContents())  # Prueba mientras no solucione lo del reward de scenario y actualizar el estado s(t+1)
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

        if self.iterations > 50000:
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
        if not self.noMotivManager:
            # self.corr_sensor, self.corr_type = self.correlationsManager.getActiveCorrelation(
            #     tuple(self.episode.getSensorialStateT1()), self.activeCorr)
            self.corr_sensor, self.corr_type = self.correlationsManager.getActiveCorrelation(
                tuple(self.simulator.get_sensorization()),
                self.activeCorr)  # Prueba, mientras no actualizo el tema del estado despues del reward de scenario
            if self.corr_sensor == 0:
                self.activeMot = 'Int'
            else:
                if self.activeMot == 'Int':
                    # self.tracesBuffer.removeAll()
                    self.iter_min = 0
                self.activeMot = 'Ext'

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
        # # Graph 1
        # plt.figure()
        # n_reward = 0
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][1]:
        #         n_reward += 1
        #         if self.graph1[i][7][1] == 0.0: # Distance Baxter-ball=0.0 when reward
        #             # plt.plot(self.graph1[i][0], self.graph1[i][2], 'ro', color='red')
        #             plt.plot(n_reward, self.graph1[i][2], 'ro', color='red')
        #         else: # The reward is given to the Robobo
        #             # plt.plot(self.graph1[i][0], self.graph1[i][2], 'ro', color='blue')
        #             plt.plot(n_reward, self.graph1[i][2], 'ro', color='blue')
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         plt.axvline(x=n_reward)
        # # for i in range(len(self.graph1)):
        # #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        # #         plt.axvline(x=self.graph1[i][0])
        #
        # # plt.axes().set_xlabel('Iterations')
        # plt.axes().set_xlabel('Number of rewards')
        # plt.axes().set_ylabel('Iterations needed to reach the goal')
        # plt.grid()
        #
        #
        # # Simple moving average
        # reward_matrix = []
        # blind_matrix= []
        # iter = []
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][1]:
        #
        #         reward_matrix.append(self.graph1[i][2])
        #         # blind_matrix.append(self.graph1[i][3])
        #         # if self.graph1[i][3] == 0: # 1 = use of Ib, 0 = no use of Ib
        #         #     blind_matrix.append(0)
        #         # else:
        #         #     blind_matrix.append(1*100)
        #         blind_matrix.append(min(self.graph1[i][3]/50, 1))
        #         iter.append(self.graph1[i][0])
        # window = 100
        # window_aux = 50
        # media = self.calcSma(reward_matrix, window)
        # media_aux = self.calcSma(reward_matrix[:window-1], window_aux)
        # media_sum = media_aux+media[window-1:]
        # # plt.plot(iter, media_sum, marker='.', color='cyan', linewidth=0.5, label='simple moving average')
        # plt.plot(range(n_reward), media_sum, marker='.', color='cyan', linewidth=0.5, label='simple moving average')
        #
        #
        # # # Accumulated reward
        # # acum = []
        # # aux = 0
        # # for i in range(len(self.graph1)):
        # #     aux += self.graph1[i][1]
        # #     acum.append(aux)
        # # plt.figure()
        # # plt.plot(range(len(acum)), acum, marker='.', color=(0.0, 0.0, 0.6), linewidth=3.0, label='accumulated reward')
        # #
        # # # Simple moving average accumulated reward
        # # window = 1500
        # # window_aux = 50
        # # media = self.calcSma(acum, window)
        # # media_aux = self.calcSma(acum[:window - 1], window_aux)
        # # media_sum = media_aux + media[window - 1:]
        # # plt.plot(range(len(media_sum)), media_sum, marker='.', color='cyan', linewidth=0.5, label='sma acumulated reward')
        # # # print len(media_sum), len(acum)
        # # plt.legend()
        # # # Rewards/iterations
        # # for i in range(len(media_sum)):
        # #     if media_sum[i]==None:
        # #         media_sum[i]=0.0
        # # ventana = 500
        # # inc = []
        # # for i in range(len(acum) / ventana - 1):
        # #     inc.append(acum[i * ventana + ventana] - acum[i * ventana])
        # # inc = [0]+inc
        # # plt.figure()
        # # plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 0.0, 0.6), linewidth=0.5, label='inc accumulated reward')
        # # ###
        # # inc = []
        # # media_sum=list(media_sum)
        # # for i in range(len(media_sum) / ventana - 1):
        # #     inc.append(media_sum[i * ventana + ventana] - media_sum[i * ventana])
        # # inc = [0]+inc
        # # plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 1.0, 0.6), linewidth=0.5, label='sma inc accumulated reward')
        # # plt.legend()
        # # #########
        # # plt.figure()
        # # acum_rew = np.array(range(len(iter) + 1))
        # # it = np.array([0] + iter)
        # # plt.plot([0] + iter, list(it/acum_rew), marker='.', color=(0.6, 0.0, 0.6), linewidth=0.5, label='iterations/accumulated reward')
        # # plt.legend()
        # # plt.figure()
        # # plt.plot([0] + iter, list(acum_rew.astype(float) / it.astype(float)), marker='.', color=(0.0, 0.6, 0.6), linewidth=0.5,
        # #          label='accumulated reward/iterations')
        # # plt.legend()
        #
        #
        # # Graph 2
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][1]:
        #         ax.plot(self.graph1[i][0], self.graph1[i][3], 'ro', color='green')
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         ax.axvline(x=self.graph1[i][0])
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Iterations the Ib motivation is active')
        # ax.grid()
        # ax2 = ax.twinx()
        # # Simple moving average: 1 = use of Ib, 0 = no use of Ib
        # window = 100
        # window_aux = 50
        # media = self.calcSma(blind_matrix, window)
        # media_aux = self.calcSma(blind_matrix[:window - 1], window_aux)
        # media_sum = media_aux + media[window - 1:]
        # ax2.plot(iter, media_sum, marker='.', color='orange', linewidth=0.5, label='active')
        # ax2.set_ylabel('Use of Ib')
        # plt.show()
        #
        # # Graph 3
        # plt.figure()
        # contS1neg = [];contS1pos = [];contS2neg = [];contS2pos = [];contS3neg = [];contS3pos = []
        # for i in range(len(self.graph2)):
        #     contS1neg.append(self.graph2[i][0])
        #     contS1pos.append(self.graph2[i][1])
        #     contS2neg.append(self.graph2[i][2])
        #     contS2pos.append(self.graph2[i][3])
        #     contS3neg.append(self.graph2[i][4])
        #     contS3pos.append(self.graph2[i][5])
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         plt.axvline(x=i+1, linewidth=2.0)
        # plt.plot(range(len(self.graph2)), contS1neg, marker='.', markersize=0.5, linewidth=0.5, color='cyan', label='S1-')
        # plt.plot(range(len(self.graph2)), contS1pos, marker='.', markersize=0.5, linewidth=0.5, color='brown', label='S1+')
        # plt.plot(range(len(self.graph2)), contS2neg, marker='.', markersize=0.5, linewidth=0.5, color='green', label='S2-')
        # plt.plot(range(len(self.graph2)), contS2pos, marker='.', markersize=0.5, linewidth=0.5, color='purple', label='S2+')
        # plt.plot(range(len(self.graph2)), contS3neg, marker='.', markersize=0.5, linewidth=0.5, color='red', label='S3-')
        # plt.plot(range(len(self.graph2)), contS3pos, marker='.', markersize=0.5, linewidth=0.5, color='orange', label='S3+')
        # plt.axes().set_xlabel('Iterations')
        # plt.axes().set_ylabel('Balance between Positive and Negative Traces')
        # plt.ylim(0, 25)
        # plt.grid()
        # plt.legend()
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
                    ax.plot(n_reward, self.graph1[i][2], 'ro', color='blue')
            if self.graph1[i][4] > self.graph1[i - 1][4]:
                ax.axvline(x=n_reward)

        # ax.axvline(x=347, linewidth=2.0, color='green', linestyle='--')
        # ax.axvline(x=363 + 5, linewidth=2.0, color='purple', linestyle='--')
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         plt.axvline(x=self.graph1[i][0])

        # plt.axes().set_xlabel('Iterations')
        ax.set_xlabel('Number of rewards')
        ax.set_ylabel('Iterations needed to reach the goal')
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(range(n_reward), iter_goal, marker='.', markersize=1.0, color='green', linewidth=1.0, label='active')
        ax2.set_ylabel('Number of iterations')
        # plt.xlim(0, 40000)
        plt.show()
        # Simple moving average
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
                    blind_matrix.append(min(self.graph1[i][3] / 50, 1))
                    iter2.append(self.graph1[i][0])
                if self.graph1[i][0] > 50000 and self.graph1[i][0] < 50770:
                    blind_matrix2.append(min(self.graph1[i][3] / 50, 1))
                    iter3.append(self.graph1[i][0])
                if self.graph1[i][0] > 50770:
                    blind_matrix3.append(min(self.graph1[i][3] / 50, 1))
                    iter4.append(self.graph1[i][0])
                iter.append(self.graph1[i][0])
        window = 100
        window_aux = 50
        media = self.calcSma(reward_matrix, window)
        media_aux = self.calcSma(reward_matrix[:window - 1], window_aux)
        media_sum = media_aux + media[window - 1:]
        # plt.plot(iter, media_sum, marker='.', color='cyan', linewidth=0.5, label='simple moving average')
        ax.plot(range(n_reward), media_sum, marker='.', markersize=2.0, color='cyan', linewidth=2.0,
                label='simple moving average')
        # print media_sum
        # print media_sum[-1]
        # print media_sum[205], media_sum[204], media_sum[206]

        # Accumulated reward
        acum = []
        aux = 0
        for i in range(len(self.graph1)):
            aux += self.graph1[i][1]
            acum.append(aux)
        plt.figure()
        plt.plot(range(len(acum)), acum, marker='.', color=(0.0, 0.0, 0.6), linewidth=3.0, label='accumulated reward')

        # Simple moving average accumulated reward
        window = 1500
        window_aux = 50
        media = self.calcSma(acum, window)
        media_aux = self.calcSma(acum[:window - 1], window_aux)
        media_sum = media_aux + media[window - 1:]
        plt.plot(range(len(media_sum)), media_sum, marker='.', color='cyan', linewidth=0.5, label='sma acumulated reward')
        plt.grid()
        # print len(media_sum), len(acum)
        plt.legend()
        # Rewards/iterations
        for i in range(len(media_sum)):
            if media_sum[i]==None:
                media_sum[i]=0.0
        ventana = 500
        inc = []
        for i in range(len(acum) / ventana - 1):
            inc.append(acum[i * ventana + ventana] - acum[i * ventana])
        inc = [0]+inc
        plt.figure()
        plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 0.0, 0.6), linewidth=0.5, label='inc accumulated reward')
        ###
        inc = []
        media_sum=list(media_sum)
        for i in range(len(media_sum) / ventana - 1):
            inc.append(media_sum[i * ventana + ventana] - media_sum[i * ventana])
        inc = [0]+inc
        plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 1.0, 0.6), linewidth=0.5, label='sma inc accumulated reward')
        plt.legend()
        plt.grid()
        #########
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
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Iterations the Ib motivation is active')
        ax.grid()
        # ax.axvline(x=50000, linewidth=2.0, color='green', linestyle='--')
        # ax.axvline(x=50483 + 287, linewidth=2.0, color='purple', linestyle='--')
        ax2 = ax.twinx()
        # Simple moving average: 1 = use of Ib, 0 = no use of Ib
        window = 100
        window_aux = 40
        media = self.calcSma(blind_matrix, window)
        media_aux = self.calcSma(blind_matrix[:window - 1], window_aux)
        media_sum = media_aux + media[window - 1:]

        # ax2.plot(iter, media_sum, marker='.', markersize=1.0, color='orange', linewidth=1.0, label='active')
        ax2.plot(iter2, media_sum, marker='.', markersize=1.0, color='orange', linewidth=1.0, label='active')

        window = 15
        window_aux = 4
        media2 = self.calcSma(blind_matrix2, window)
        media_aux2 = self.calcSma(blind_matrix2[:window - 1], window_aux)
        media_sum2 = media_aux2 + media2[window - 1:]

        ax2.plot(iter3, media_sum2, marker='.', markersize=1.0, color='orange', linewidth=1.0, label='active')

        window = 10
        window_aux = 4
        media3 = self.calcSma(blind_matrix3, window)
        media_aux3 = self.calcSma(blind_matrix3[:window - 1], window_aux)
        media_sum3 = media_aux3 + media3[window - 1:]

        ax2.plot(iter4, media_sum3, marker='.', markersize=1.0, color='orange', linewidth=2.0, label='active')

        ax2.set_ylabel('Use of Ib')
        # plt.xlim(0, 40000)
        plt.show()

    def plotScaffoldingMap(self):

        f = open('scaffoldingMatrixLongExec.pckl', 'rb')
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
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'orange' and color != 'blue':
                        ax.scatter(i, j, k, c=color, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Distance Robobo-cylinder')
        ax.set_ylabel('Distance Baxter-cylinder')
        ax.set_zlabel('Distance basket-cylinder')

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
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'blue':
                        ax.scatter(i, j, k, c=color, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Distance Robobo-cylinder')
        ax.set_ylabel('Distance Baxter-cylinder')
        ax.set_zlabel('Distance basket-cylinder')

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
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red':
                        ax.scatter(i, j, k, c=color, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Distance Robobo-cylinder')
        ax.set_ylabel('Distance Baxter-cylinder')
        ax.set_zlabel('Distance basket-cylinder')

        plt.show()

        # Subfigure 1
        plt.figure()
        ax1 = plt.subplot(241)
        ax1.set_xlim(0, len(ScaffoldingMatrix))
        ax1.set_ylim(0, len(ScaffoldingMatrix))
        ax1.set_xlabel('distance ball-Robobo')
        ax1.set_ylabel('distance ball-box')
        ax1.set_title("dbB = 0")
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][0][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][0][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][0][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][0][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax1.scatter(i, j, c=color, marker='s', s=50, linewidth=1.0)
        # # Subfigure 2
        ax2 = plt.subplot(242)
        ax2.set_xlim(0, len(ScaffoldingMatrix))
        ax2.set_ylim(0, len(ScaffoldingMatrix))
        ax2.set_title("dbB = 500")
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][5][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][5][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][5][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][5][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax2.scatter(i, j, c=color, marker='s', s=30, linewidth=0)
        # # Subfigure 3
        ax3 = plt.subplot(243)
        # ax3.axis('off')
        ax3.set_xlim(0, len(ScaffoldingMatrix))
        ax3.set_ylim(0, len(ScaffoldingMatrix))
        ax3.set_title("dbB = 1000")
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][10][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][10][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][10][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][10][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax3.scatter(i, j, c=color, marker='s', s=30, linewidth=0)
        # Subfigure 4
        ax4 = plt.subplot(244)
        # ax4.axis('off')
        ax4.set_xlim(0, len(ScaffoldingMatrix))
        ax4.set_ylim(0, len(ScaffoldingMatrix))
        ax4.set_title("dbB = 1373")
        ax4.set_xlabel('distance ball-Robobo')
        # ax4.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][-1][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][-1][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][-1][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][-1][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax4.scatter(i, j, c=color, marker='s', s=30, linewidth=0)

        # Subfigure 5
        ax5 = plt.subplot(245)
        ax5.set_xlim(0, len(ScaffoldingMatrix))
        ax5.set_ylim(0, len(ScaffoldingMatrix))
        ax5.set_title("dbR = 0")
        ax5.set_xlabel('distance ball-Baxter')
        ax5.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[0][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[0][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[0][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[0][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax5.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        # Subfigure 6
        ax6 = plt.subplot(246)
        # ax6.axis('off')
        ax6.set_xlim(0, len(ScaffoldingMatrix))
        ax6.set_ylim(0, len(ScaffoldingMatrix))
        ax6.set_title("dbR = 500")
        # ax6.set_xlabel('distance ball-Baxter')
        # ax6.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[5][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[5][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[5][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[5][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax6.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        # Subfigure 7
        ax7 = plt.subplot(247)
        # ax7.axis('off')
        ax7.set_xlim(0, len(ScaffoldingMatrix))
        ax7.set_ylim(0, len(ScaffoldingMatrix))
        ax7.set_title("dbR = 1000")
        # ax7.set_xlabel('distance ball-Baxter')
        # ax7.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[10][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[10][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[10][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[10][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax7.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        # Subfigure 8
        ax8 = plt.subplot(248)
        # ax8.axis('off')
        ax8.set_xlim(0, len(ScaffoldingMatrix))
        ax8.set_ylim(0, len(ScaffoldingMatrix))
        ax8.set_title("dbR = 1373")
        ax8.set_xlabel('distance ball-Baxter')
        # ax8.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[-1][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[-1][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[-1][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[-1][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                ax8.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        plt.draw()

##############################
        # Subfigure 1
        plt.figure()
        ax1 = plt.subplot(241)
        ax1.set_xlim(0, len(ScaffoldingMatrix))
        ax1.set_ylim(0, len(ScaffoldingMatrix))
        ax1.set_xlabel('distance ball-Robobo')
        ax1.set_ylabel('distance ball-box')
        ax1.set_title("dbB = 0")
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][0][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][0][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][0][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][0][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax1.scatter(i, j, c=color, marker='s', s=50, linewidth=1.0)
        # # Subfigure 2
        ax2 = plt.subplot(242)
        # ax2.axis('off')
        ax2.set_xlim(0, len(ScaffoldingMatrix))
        ax2.set_ylim(0, len(ScaffoldingMatrix))
        ax2.set_title("dbB = 500")
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][5][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][5][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][5][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][5][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax2.scatter(i, j, c=color, marker='s', s=30, linewidth=0)
        # # Subfigure 3
        ax3 = plt.subplot(243)
        # ax3.axis('off')
        ax3.set_xlim(0, len(ScaffoldingMatrix))
        ax3.set_ylim(0, len(ScaffoldingMatrix))
        ax3.set_title("dbB = 1000")
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][10][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][10][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][10][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][10][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax3.scatter(i, j, c=color, marker='s', s=30, linewidth=0)
        # Subfigure 4
        ax4 = plt.subplot(244)
        # ax4.axis('off')
        ax4.set_xlim(0, len(ScaffoldingMatrix))
        ax4.set_ylim(0, len(ScaffoldingMatrix))
        ax4.set_title("dbB = 1373")
        ax4.set_xlabel('distance ball-Robobo')
        # ax4.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[i][-1][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[i][-1][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[i][-1][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[i][-1][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax4.scatter(i, j, c=color, marker='s', s=30, linewidth=0)

        # Subfigure 5
        ax5 = plt.subplot(245)
        ax5.set_xlim(0, len(ScaffoldingMatrix))
        ax5.set_ylim(0, len(ScaffoldingMatrix))
        ax5.set_title("dbR = 0")
        ax5.set_xlabel('distance ball-Baxter')
        ax5.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[0][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[0][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[0][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[0][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax5.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        # Subfigure 6
        ax6 = plt.subplot(246)
        # ax6.axis('off')
        ax6.set_xlim(0, len(ScaffoldingMatrix))
        ax6.set_ylim(0, len(ScaffoldingMatrix))
        ax6.set_title("dbR = 500")
        # ax6.set_xlabel('distance ball-Baxter')
        # ax6.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[5][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[5][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[5][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[5][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax6.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        # Subfigure 7
        ax7 = plt.subplot(247)
        # ax7.axis('off')
        ax7.set_xlim(0, len(ScaffoldingMatrix))
        ax7.set_ylim(0, len(ScaffoldingMatrix))
        ax7.set_title("dbR = 1000")
        # ax7.set_xlabel('distance ball-Baxter')
        # ax7.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[10][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[10][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[10][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[10][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax7.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        # Subfigure 8
        ax8 = plt.subplot(248)
        # ax8.axis('off')
        ax8.set_xlim(0, len(ScaffoldingMatrix))
        ax8.set_ylim(0, len(ScaffoldingMatrix))
        ax8.set_title("dbR = 1373")
        ax8.set_xlabel('distance ball-Baxter')
        # ax8.set_ylabel('distance ball-box')
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                # Establezco el color del punto en funcion del valor de certeza
                if ScaffoldingMatrix[-1][i][j] == 0.0:
                    color = 'red'
                elif ScaffoldingMatrix[-1][i][j] == 1.0:
                    color = 'green'
                elif ScaffoldingMatrix[-1][i][j] == 2.0:
                    color = 'orange'
                elif ScaffoldingMatrix[-1][i][j] == 3.0:
                    color = 'blue'
                else:
                    color = 'white'
                # Dibujo el punto
                if color != 'red':
                    ax8.scatter(i, j, c=color, marker='o', s=30, linewidth=0)

        plt.draw()

    def calcSma(sel, data, smaPeriod):
        j = next(i for i, x in enumerate(data) if x is not None)
        our_range = range(len(data))[j + smaPeriod - 1:]
        empty_list = [None] * (j + smaPeriod - 1)
        sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

        return list(empty_list + sub_result)

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
            ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
            plt.show()

    def plotEpisodes(self, graph, Lsup, Linf):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('Distance Robobo - Cylinder')
        ax.set_ylabel('Distance Baxter - Cylinder')
        ax.set_zlabel('Distance Basket - Cylinder')
        plt.title('Representative executions from ' + str(Linf) + ' to ' + str(Lsup))
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

    def saveData(self):  # , action, seed):
        f = open('SimulationDataLongExecTraces.pckl', 'wb')
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

    def loadData(self, name):
        f = open(str(name)+'.pckl', 'rb')

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
        # self.graph2 = pickle.load(f)
        # self.graph3 = pickle.load(f)
        # self.graph4 = pickle.load(f)
        # self.graph5 = pickle.load(f)
        # self.graph6 = pickle.load(f)
        # self.graph7 = pickle.load(f)
        self.TracesMemoryVF = pickle.load(f)
        f.close()

    def saveScaffoldingMatrix(self):
        '''Ad hoc for the paper'''
        Map1 = self.correlationsManager.correlations[0].S3_neg.getCertaintyMatrix()
        Map2 = self.correlationsManager.correlations[1].S2_neg.getCertaintyMatrix()
        Map3 = self.correlationsManager.correlations[2].S1_neg.getCertaintyMatrix()

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
        print ScaffoldingMatrix

        f = open('scaffoldingMatrix.pckl', 'wb')
        pickle.dump(ScaffoldingMatrix, f)
        f.close()

    def plotMediaGraphs(self):

        # Calculate Data
        # Calcular media y append()
        f = open('GraphMediaData.pckl', 'rb')
        graph1_1 = pickle.load(f)
        graph1_2 = pickle.load(f)
        graph1_3 = pickle.load(f)
        graph1_4 = pickle.load(f)
        graph1_5 = pickle.load(f)
        graph1_6 = pickle.load(f)
        graph1_7 = pickle.load(f)
        graph1_8 = pickle.load(f)
        graph1_9 = pickle.load(f)
        graph1_10 = pickle.load(f)
        f.close()

        # Calculo rewards acumulado para cada uno de ellos
        acum1 = self.CalculaAcumulado(graph1_1)
        acum2 = self.CalculaAcumulado(graph1_2)
        acum3 = self.CalculaAcumulado(graph1_3)
        acum4 = self.CalculaAcumulado(graph1_4)
        acum5 = self.CalculaAcumulado(graph1_5)
        acum6 = self.CalculaAcumulado(graph1_6)
        acum7 = self.CalculaAcumulado(graph1_7)
        acum8 = self.CalculaAcumulado(graph1_8)
        acum9 = self.CalculaAcumulado(graph1_9)
        acum10 = self.CalculaAcumulado(graph1_10)

        # Plot acumulated rewards
        plt.figure()
        plt.plot(range(len(acum1)), acum1, marker='.', color='olive', linewidth=0.5, markersize=0.5, label='Exec. 1')
        plt.plot(range(len(acum2)), acum2, marker='.', color='green', linewidth=.5, markersize=0.5, label='Exec. 2')
        plt.plot(range(len(acum3)), acum3, marker='.', color='red', linewidth=0.5, markersize=0.5, label='Exec. 3')
        plt.plot(range(len(acum4)), acum4, marker='.', color='pink', linewidth=0.5, markersize=0.5, label='Exec. 4')
        plt.plot(range(len(acum5)), acum5, marker='.', color='orange', linewidth=0.5, markersize=0.5, label='Exec. 5')
        plt.plot(range(len(acum6)), acum6, marker='.', color='purple', linewidth=0.5, markersize=0.5, label='Exec. 6')
        plt.plot(range(len(acum7)), acum7, marker='.', color='yellow', linewidth=0.5, markersize=0.5, label='Exec. 7')
        plt.plot(range(len(acum8)), acum8, marker='.', color='cyan', linewidth=0.5, markersize=0.5, label='Exec. 8')
        plt.plot(range(len(acum9)), acum9, marker='.', color='grey', linewidth=0.5, markersize=0.5, label='Exec. 9')
        plt.plot(range(len(acum10)), acum10, marker='.', color='brown', linewidth=0.5, markersize=0.5, label='Exec. 10')
        plt.grid()



        # Calculo media
        acumMedia=[]
        for i in range(len(graph1_1)):
            # media = (acum1[i] + acum2[i] + acum3[i] + acum4[i] + acum5[i] + acum6[i] + acum7[i] + acum8[i] + acum9[i] + acum10[i]) / 10
            media = (acum1[i] + acum2[i] + acum3[i] + acum4[i] + acum5[i] + acum6[i] + acum7[i] + acum9[i]) / 8.0
            acumMedia.append(media)

        plt.plot(range(len(acumMedia)), acumMedia, marker='x', color='blue', linewidth=0.5, markersize=3.0, label='Average value')
        # plt.legend(loc='best')

        # Simple moving average accumulated reward
        window = 1500
        window_aux = 50
        media = self.calcSma(acumMedia, window)
        media_aux = self.calcSma(acumMedia[:window - 1], window_aux)
        media_sum = media_aux + media[window - 1:]
        # plt.plot(range(len(media_sum)), media_sum, marker='.', color='cyan', linewidth=0.5,label='sma acumulated reward')
        plt.legend(loc='best')
        plt.title('Series of representative executions', size=15.0)
        plt.ylabel('Number of rewards', size=15.0)
        plt.xlabel('Iterations', size=15.0)



        # ventana = 500
        # inc = []
        # for i in range(len(acumMedia) / ventana - 1):
        #     inc.append(acumMedia[i * ventana + ventana] - acumMedia[i * ventana])
        # inc = [0]+inc
        # plt.figure()
        # plt.plot(np.array(range(len(inc)))*ventana, inc, marker='.', color=(0.0, 0.0, 0.6), linewidth=0.5, label='inc accumulated reward')

        # Rewards/iterations
        for i in range(len(media_sum)):
            if media_sum[i] == None:
                media_sum[i] = 0.0
        ventana = 3000
        inc = []
        for i in range(len(acumMedia) / ventana - 1):
            inc.append(acumMedia[i * ventana + ventana] - acumMedia[i * ventana])
        inc = [0] + inc
        plt.figure()
        plt.plot(np.array(range(len(inc))) * ventana, inc, marker='.', color=(0.0, 0.0, 0.6), linewidth=0.5,
                 label='inc accumulated reward')
        ###
        inc = []
        media_sum = list(media_sum)
        for i in range(len(media_sum) / ventana - 1):
            inc.append(media_sum[i * ventana + ventana] - media_sum[i * ventana])
        inc = [0] + inc
        plt.plot(np.array(range(len(inc))) * ventana, inc, marker='.', markersize=2.0, color=(0.0, 1.0, 0.6), linewidth=2.0,
                 label='sma inc accumulated reward')
        plt.legend(loc='best')
        plt.grid()

    def CalculaAcumulado(self, matrix):
        acum = []
        aux = 0
        for i in range(len(matrix)):
            aux += matrix[i][1]
            acum.append(aux)

        return acum


def main():
    instance = MDBCore()
    instance.run()


if __name__ == '__main__':
    main()


# import pickle;f=open('store.pckl', 'rb');seed=pickle.load(f);f.close();import numpy as np;np.random.set_state(seed);from MDBCore import *;a=MDBCore()
            # a.run()
# seed=np.random.get_state();import pickle;f=open('seed_object.pckl', 'wb');pickle.dump(seed,f);f.close()

