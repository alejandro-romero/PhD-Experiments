from Simulador import *
from Episode import *
from CandidateStateEvaluator import *
from TracesBuffer import *
from CorrelationsManager import *
import logging
import pickle

class MDBCore(object):
    def __init__(self):

        self.memoryVF = TracesBuffer()
        self.memoryVF.setMaxSize(20)
        self.TracesMemoryVF = TracesMemory()
        # Object initialization
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

        self.probUseGuided = 0  # Probability of using Intrinsic guided motivation during the Extrinsic use
        self.useVF = 0  # Indicate which Utility Model use (VF=1 or SUR=0) when Extrinsic motivation is active
        self.n_new_traces = 1001

    def run(self):


        # Save/load seed

        # # Import seed
        f = open('seed_HAIS.pckl', 'rb')
        seed = pickle.load(f)
        f.close()
        np.random.set_state(seed)

        #Save seed
        # seed = np.random.get_state()
        # f = open('seed_HAIS.pckl', 'wb')
        # pickle.dump(seed, f)
        # f.close()

        self.main()

        # Save/close logs

    def main(self):

        self.stop = 0
        self.iterations = 0

        self.loadData()
        self.iterations = 12001
        action = self.CSE.actionChooser.getCandidateActions()[0]
        self.activeMot='Ext'

        while not self.stop:

            if self.iterations == 0:
                action = self.CSE.actionChooser.getCandidateActions()[0]

            # Sensorization in t (distances, action and motivation)
            self.episode.setSensorialStateT(self.simulator.get_sensorization())
            self.episode.setAction(action)
            # self.episode.setMotivation(self.activeMot)

            self.simulator.baxter_larm_action(action)

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

            # Check if a new correlation is needed
            self.correlationsManager.newCorrelation()
            if self.correlationsManager.correlations[self.activeCorr].i_reward_assigned == 0:
                self.correlationsManager.assignRewardAssigner(self.activeCorr, self.episode.getSensorialStateT1())

            # Memory Manager (Traces, weak traces and antitraces)
            if self.activeMot == 'Int':
                self.it_blind += 1
                self.noMotivManager = 0
                # If there is a reward, realise reward assignment and save trace in Traces Memory
                if self.episode.getReward():
                    if self.correlationsManager.correlations[self.activeCorr].i_reward_assigned == 0:
                        self.correlationsManager.correlations[self.activeCorr].i_reward_assigned = 1
                        self.correlationsManager.correlations[self.activeCorr].i_reward = None

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
                ##########
                if self.CSE.intrinsicGuidedActive:
                    self.noMotivManager = 1
                    # Miro la certeza y en el momento en que deje de ser mayor que el umbral,
                    # activo la motivacion extrinseca con la correlacion que habia activa antes
                    corr_sensor_comp, corr_type_comp = self.correlationsManager.correlations[self.activeCorr].getActiveCorrelation(tuple(self.episode.getSensorialStateT1()))

                    if (corr_sensor_comp != self.corr_sensor or corr_type_comp != self.corr_type) and (not self.CSE.followOriginalCorrelation):
                        # Sigo la correlacion que estaba activa antes
                        self.CSE.followOriginalCorrelation = 1
                        self.correlationsManager.correlations[self.activeCorr].addIgTrace(self.tracesBuffer.getTrace(),
                                                                                        self.corr_sensor, self.corr_type, 1)
                        self.reinitializeMemories()

                    if self.episode.getReward():  # GOAL MANAGER - Encargado de asignar la recompensa?
                        self.simulator.restart_scenario()  # Restart scenario
                        # Save as trace in TracesMemory of the correlated sensor
                        self.correlationsManager.correlations[self.activeCorr].addTrace(self.tracesBuffer.getTrace(),
                                                                                        self.corr_sensor, self.corr_type, 1)
                        self.CSE.intrinsicGuidedActive = 0
                        self.CSE.followOriginalCorrelation = 0
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

                        self.TracesMemoryVF.addTraces(self.memoryVF.getTraceReward())
                        self.memoryVF.removeAll()

                    elif self.correlationsManager.getReward(self.activeCorr, self.simulator.get_reward(),
                                                            tuple(self.episode.getSensorialStateT1())):
                        # Save as trace in TracesMemory of the correlated sensor
                        self.correlationsManager.correlations[self.activeCorr].addTrace(self.tracesBuffer.getTrace(),
                                                                                        self.corr_sensor, self.corr_type, 1)
                        self.CSE.intrinsicGuidedActive = 0
                        self.CSE.followOriginalCorrelation = 0
                        # The active correlation is now the correlation that has provided the reward
                        self.activeCorr = self.correlationsManager.correlations[self.activeCorr].i_reward
                        self.reinitializeMemories()
                        logging.info('Correlation reward when Extrinsic Motivation')

                        self.noMotivManager = 0
                    else:
                        # Check if the the active correlation is still active
                        if self.iter_min > 2:
                            if self.CSE.followOriginalCorrelation:
                                sens_t = self.tracesBuffer.getTrace()[-2][self.corr_sensor - 1]
                                sens_t1 = self.tracesBuffer.getTrace()[-1][self.corr_sensor - 1]
                            else:
                                sens_t = self.tracesBuffer.getTrace()[-2][self.CSE.corr_sensor_new - 1]
                                sens_t1 = self.tracesBuffer.getTrace()[-1][self.CSE.corr_sensor_new - 1]


                            dif = sens_t1 - sens_t

                            if (self.corr_type == 'pos' and dif <= 0) or (self.corr_type == 'neg' and dif >= 0):
                                # Guardo antitraza en el sensor correspondiente y vuelvo a comezar el bucle
                                self.correlationsManager.correlations[self.activeCorr].addAntiTrace(
                                    self.tracesBuffer.getTrace(), self.corr_sensor, self.corr_type, 1)

                                self.CSE.intrinsicGuidedActive = 0
                                self.CSE.followOriginalCorrelation = 0
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

                else:
                    #######
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

                            corr_still_active = 1
                            if self.correlationsManager.correlations[self.activeCorr].established:
                                corr_still_active = self.correlationsManager.correlations[self.activeCorr].getActiveCorrelation(self.simulator.get_sensorization())[1]
                                # if corr_still_active == '':  # La certeza de la correelacion es menor que el umbral
                                #     self.noMotivManager = 0

                            sens_t = self.tracesBuffer.getTrace()[-2][self.corr_sensor - 1]
                            sens_t1 = self.tracesBuffer.getTrace()[-1][self.corr_sensor - 1]
                            dif = sens_t1 - sens_t

                            if (self.corr_type == 'pos' and dif <= 0) or (self.corr_type == 'neg' and dif >= 0) or corr_still_active == '':
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
                self.simulator.ball_get_pos(),
                self.simulator.ball_position, self.simulator.balise1_get_pos(), self.simulator.balise2_get_pos())
            # action = self.CSE.getAction(self.activeMot, SimData, tuple(self.episode.getSensorialStateT1()),
            #                             self.corr_sensor, self.corr_type, self.intrinsicMemory.getContents())

            if self.corr_sensor == 0:
                if self.corr_type == 'pos':
                    Tb = self.correlationsManager.correlations[self.activeCorr].S1_pos.numberOfGoalsWithoutAntiTraces
                else:
                    Tb = self.correlationsManager.correlations[self.activeCorr].S1_neg.numberOfGoalsWithoutAntiTraces
            elif self.corr_sensor == 1:
                if self.corr_type == 'pos':
                    Tb = self.correlationsManager.correlations[self.activeCorr].S2_pos.numberOfGoalsWithoutAntiTraces
                else:
                    Tb = self.correlationsManager.correlations[self.activeCorr].S2_neg.numberOfGoalsWithoutAntiTraces
            else:
                Tb = 0


            maxTb = self.correlationsManager.correlations[self.activeCorr].Tb_max

            # Usar VF a partir de las 10000 iteraciones (como primera aproximacion rapida)
            if self.iterations > 10000:
                self.useVF = 1

            # if self.episode.getReward() and self.n_new_traces > 1000:
            if self.n_new_traces > 1000:
                trainNet=1
                self.n_new_traces = 0
            else:
                trainNet=0
                if self.episode.getReward():
                    self.n_new_traces += 1

            action = self.CSE.getAction(self.activeMot, SimData, tuple(self.simulator.get_sensorization()),
                                            self.corr_sensor, self.corr_type,
                                            self.intrinsicMemory.getContents(), Tb, maxTb,
                                            self.correlationsManager.correlations[self.activeCorr].established,
                                            self.probUseGuided, self.useVF, self.TracesMemoryVF.getTracesList(), trainNet)


            # Others
            # self.writeLogs()
            self.debugPrint()
            self.iter_min += 1
            self.iterations += 1
            self.it_reward += 1
            self.stopCondition()
            self.episode.cleanEpisode()

        self.saveData()

    def stopCondition(self):

        if self.iterations > 12500:#10000:
            self.stop = 1

    def writeLogs(self):
        logging.debug('%s  -  %s  -  %s  -  %s  -  %s  -  %s', self.iterations, self.activeMot, self.activeCorr,
                      self.corr_sensor, self.corr_type, self.episode.getEpisode())

    def debugPrint(self):
        print '------------------'
        print "Iteration: ", self.iterations
        print "Active correlation: ", self.activeCorr
        print "Active motivation: ", self.activeMot
        print "Correlated sensor: ", self.corr_sensor, self.corr_type
        print "Trazas consecutivas S1 neg: ", self.correlationsManager.correlations[
            self.activeCorr].S1_neg.numberOfGoalsWithoutAntiTraces
        print "Trazas consecutivas S1 pos: ", self.correlationsManager.correlations[
            self.activeCorr].S1_pos.numberOfGoalsWithoutAntiTraces
        print "Trazas consecutivas S2 neg: ", self.correlationsManager.correlations[
            self.activeCorr].S2_neg.numberOfGoalsWithoutAntiTraces
        print "Trazas consecutivas S2 pos: ", self.correlationsManager.correlations[
            self.activeCorr].S2_pos.numberOfGoalsWithoutAntiTraces
        # print "Trazas consecutivas S3 neg: ", self.correlationsManager.correlations[
        #     self.activeCorr].S3_neg.numberOfGoalsWithoutAntiTraces
        # print "Trazas consecutivas S3 pos: ", self.correlationsManager.correlations[
        #     self.activeCorr].S3_pos.numberOfGoalsWithoutAntiTraces

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
                self.probUseGuided = np.random.choice([0, 1], p=[1, 0])

    def saveGraphs(self):
        # Graph 1 - Iterations to reach the goal vs. Total number of iterations
        self.graph1.append(
            (self.iterations, self.episode.getReward(), self.it_reward, self.it_blind,
             len(self.correlationsManager.correlations), self.activeMot, self.activeCorr,
             self.episode.getSensorialStateT1()))

        # self.graph2.append((self.correlationsManager.correlations[-1].S1_neg.numberOfGoalsWithoutAntiTraces,
        #                    self.correlationsManager.correlations[-1].S1_pos.numberOfGoalsWithoutAntiTraces,
        #                    self.correlationsManager.correlations[-1].S2_neg.numberOfGoalsWithoutAntiTraces,
        #                    self.correlationsManager.correlations[-1].S2_pos.numberOfGoalsWithoutAntiTraces,
        #                    self.correlationsManager.correlations[-1].S3_neg.numberOfGoalsWithoutAntiTraces,
        #                    self.correlationsManager.correlations[-1].S3_pos.numberOfGoalsWithoutAntiTraces))

        self.graph2.append((self.correlationsManager.correlations[-1].S1_neg.numberOfGoalsWithoutAntiTraces,
                           self.correlationsManager.correlations[-1].S1_pos.numberOfGoalsWithoutAntiTraces,
                           self.correlationsManager.correlations[-1].S2_neg.numberOfGoalsWithoutAntiTraces,
                           self.correlationsManager.correlations[-1].S2_pos.numberOfGoalsWithoutAntiTraces))

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
        n_reward = 0
        ax = fig.add_subplot(111)
        iter_goal = []  # Number of iteration increase at the same time as goals
        for i in range(11000):# for i in range(len(self.graph1)):#for i in range(len(self.graph1)):
            if self.graph1[i][1]:
                n_reward += 1
                iter_goal.append(self.graph1[i][0])
                if self.graph1[i][7][1] == 0.0: # Distance Baxter-ball=0.0 when reward
                    # plt.plot(self.graph1[i][0], self.graph1[i][2], 'ro', color='red')
                    ax.plot(n_reward, self.graph1[i][2], 'ro', color='red')
                else: # The reward is given to the Robobo
                    # plt.plot(self.graph1[i][0], self.graph1[i][2], 'ro', color='blue')
                    ax.plot(n_reward, self.graph1[i][2], 'ro', color='blue')
            if self.graph1[i][4] > self.graph1[i - 1][4]:  # Marco creacion nuevas SUR
                ax.axvline(x=n_reward)
            if i == 9396:
                ax.axvline(x=n_reward)

        ax.axvline(x=224.5, linestyle='--', color='violet', linewidth=2.0)
        # for i in range(len(self.graph1)):
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         plt.axvline(x=self.graph1[i][0])

        # plt.axes().set_xlabel('Iterations')
        ax.set_xlabel('Goal achievements', fontsize=15.0)
        ax.set_ylabel('Iterations needed to reach the goal', fontsize=15.0)
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(range(n_reward), iter_goal, marker='.', markersize=1.0, color='green', linewidth=1.0, label='active')
        ax2.set_ylabel('Number of iterations', fontsize=15.0)
        # Simple moving average
        reward_matrix = []
        blind_matrix= []
        iter = []
        for i in range(len(self.graph1)):
        # for i in range(len(self.graph1)):
            if self.graph1[i][1]:

                reward_matrix.append(self.graph1[i][2])
                # blind_matrix.append(self.graph1[i][3])
                # if self.graph1[i][3] == 0: # 1 = use of Ib, 0 = no use of Ib
                #     blind_matrix.append(0)
                # else:
                #     blind_matrix.append(1*100)
                blind_matrix.append(min(self.graph1[i][3]/50, 1))
                iter.append(self.graph1[i][0])
        window = 100
        window_aux = 50
        media = self.calcSma(reward_matrix, window)
        media_aux = self.calcSma(reward_matrix[:window-1], window_aux)
        media_sum = media_aux+media[window-1:]
        # plt.plot(iter, media_sum, marker='.', color='cyan', linewidth=0.5, label='simple moving average')
        # ax.plot(range(n_reward), media_sum, marker='.', color='cyan', linewidth=0.5, label='simple moving average')

        # Graph 2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.graph1)):# for i in range(len(self.graph1)):
            if self.graph1[i][1]:
                ax.plot(self.graph1[i][0], self.graph1[i][3], 'ro', color='green')
        # for i in range(93000):#for i in range(len(self.graph1)):
            if self.graph1[i][4] > self.graph1[i - 1][4]:
                ax.axvline(x=self.graph1[i][0])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Iterations the Ib motivation is active')
        ax.grid()
        ax2 = ax.twinx()
        # Simple moving average: 1 = use of Ib, 0 = no use of Ib
        window = 100
        window_aux = 30
        media = self.calcSma(blind_matrix, window)
        media_aux = self.calcSma(blind_matrix[:window - 1], window_aux)
        media_sum = media_aux + media[window - 1:]
        ax2.plot(iter, media_sum, marker='.', color='orange', linewidth=0.5, label='active')
        ax2.set_ylabel('Use of Ib')
        ax2.set_ylim(0, 1)
        plt.show()

        # Graph 3
        plt.figure()
        contS1neg = [];contS1pos = [];contS2neg = [];contS2pos = [];#contS3neg = [];contS3pos = []
        for i in range(len(self.graph2)):
            contS1neg.append(self.graph2[i][0])
            contS1pos.append(self.graph2[i][1])
            contS2neg.append(self.graph2[i][2])
            contS2pos.append(self.graph2[i][3])
            # contS3neg.append(self.graph2[i][4])
            # contS3pos.append(self.graph2[i][5])
            if self.graph1[i][4] > self.graph1[i - 1][4]:
                plt.axvline(x=i+1, linewidth=2.0)
        plt.plot(range(len(self.graph2)), contS1neg, marker='.', markersize=0.5, linewidth=0.5, color='cyan', label='S1-')
        plt.plot(range(len(self.graph2)), contS1pos, marker='.', markersize=0.5, linewidth=0.5, color='brown', label='S1+')
        plt.plot(range(len(self.graph2)), contS2neg, marker='.', markersize=0.5, linewidth=0.5, color='green', label='S2-')
        plt.plot(range(len(self.graph2)), contS2pos, marker='.', markersize=0.5, linewidth=0.5, color='purple', label='S2+')
        # plt.plot(range(len(self.graph2)), contS3neg, marker='.', markersize=0.5, linewidth=0.5, color='red', label='S3-')
        # plt.plot(range(len(self.graph2)), contS3pos, marker='.', markersize=0.5, linewidth=0.5, color='orange', label='S3+')
        # plt.plot(range(93000), contS1neg, marker='.', markersize=0.5, linewidth=0.5, color='cyan', label='S1-')
        # plt.plot(range(93000), contS1pos, marker='.', markersize=0.5, linewidth=0.5, color='brown', label='S1+')
        # plt.plot(range(93000), contS2neg, marker='.', markersize=0.5, linewidth=0.5, color='green', label='S2-')
        # plt.plot(range(93000), contS2pos, marker='.', markersize=0.5, linewidth=0.5, color='purple', label='S2+')
        # plt.plot(range(93000), contS3neg, marker='.', markersize=0.5, linewidth=0.5, color='red', label='S3-')
        # plt.plot(range(93000), contS3pos, marker='.', markersize=0.5, linewidth=0.5, color='orange', label='S3+')
        plt.axes().set_xlabel('Iterations')
        plt.axes().set_ylabel('Balance between Positive and Negative Traces')
        plt.ylim(0, 30)
        plt.grid()
        plt.legend()

        ## Graph 4
        # S1 = []
        # S2 = []
        # S3 = []
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # Linf = 46500
        # Lsup = 47000
        # for i in range(Linf, Lsup):
        #     S1.append(self.graph1[i][7][0])
        #     S2.append(self.graph1[i][7][1])
        #     # S3.append(self.graph1[i][7][2])
        #     if self.graph1[i][5] == 'Int':
        #         if self.graph1[i][1]:
        #             ax.plot(self.graph1[i][0], 0, 'ro', color='cyan')
        #         else:
        #             ax.plot(self.graph1[i][0], 0, 'ro', color='green')
        #     elif self.graph1[i][5] == 'Ext':
        #         if self.graph1[i][1]:
        #             ax.plot(self.graph1[i][0], self.graph1[i][6] + 1, 'ro', color='cyan')
        #         else:
        #             ax.plot(self.graph1[i][0], self.graph1[i][6]+1, 'ro', color='red')
        #     if self.graph1[i][1]:
        #         ax.axvline(x=i, ls='--', color='cyan')
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         ax.axvline(x=i)
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Active correlation')
        # ax.set_ylim(0, 12)
        # ax.grid()
        # ax2 = ax.twinx()
        # ax2.plot(range(Linf, Lsup), S1, marker='.', color='orange', linewidth=0.5, label='active')
        # ax2.plot(range(Linf, Lsup), S2, marker='.', color='grey', linewidth=0.5, label='active')
        # # ax2.plot(range(Linf, Lsup), S3, marker='.', color='pink', linewidth=0.5, label='active')
        # ax2.set_ylabel('Distances')
        # # ax2.set_ylim(0, 1)
        # plt.show()
        #
        # # Graph 5
        # S1 = []
        # S2 = []
        # S3 = []
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # Linf = 31500
        # Lsup = 34200
        # for i in range(Linf, Lsup):
        #     S1.append(self.graph1[i][7][0])
        #     S2.append(self.graph1[i][7][1])
        #     # S3.append(self.graph1[i][7][2])
        #     if self.graph1[i][5] == 'Int':
        #         if self.graph1[i][1]:
        #             ax.plot(self.graph1[i][0], 0, 'ro', color='cyan')
        #         else:
        #             ax.plot(self.graph1[i][0], 0, 'ro', color='green')
        #     elif self.graph1[i][5] == 'Ext':
        #         if self.graph1[i][1]:
        #             ax.plot(self.graph1[i][0], self.graph1[i][6] + 1, 'ro', color='cyan')
        #         else:
        #             ax.plot(self.graph1[i][0], self.graph1[i][6] + 1, 'ro', color='red')
        #     if self.graph1[i][1]:
        #         ax.axvline(x=i, ls='--', color='cyan')
        #     if self.graph1[i][4] > self.graph1[i - 1][4]:
        #         ax.axvline(x=i)
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Active correlation')
        # ax.set_ylim(0, 12)
        # ax.grid()
        # ax2 = ax.twinx()
        # ax2.plot(range(Linf, Lsup), S1, marker='.', color='orange', linewidth=0.5, label='DBaxtBall')
        # ax2.plot(range(Linf, Lsup), S2, marker='.', color='grey', linewidth=0.5, label='DBaxtB1')
        # # ax2.plot(range(Linf, Lsup), S3, marker='.', color='pink', linewidth=0.5, label='DBaxtB2')
        # ax2.set_ylabel('Distances')
        # # # ax2.set_ylim(0, 1)
        # plt.show()
        # plt.legend()


    def plotScaffoldingMap(self):

        f = open('scaffoldingMatrixSURs.pckl', 'rb')
        ScaffoldingMatrix = pickle.load(f)
        f.close()

        # print ScaffoldingMatrix
        # Correlation 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'orange' and color != 'blue' and color != 'cyan' and color != 'white':
                        ax.scatter(i, j, c=color, linewidth=0)#, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Sensor1')
        ax.set_ylabel('Sensor2')
        plt.show()
        # Correlation 1+2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'blue' and color != 'cyan' and color != 'white':
                        ax.scatter(i, j, c=color, linewidth=0)#, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Sensor1')
        ax.set_ylabel('Sensor2')
        plt.show()
        # Correlation 1+2+3
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'cyan' and color != 'white':
                        ax.scatter(i, j, c=color, linewidth=0)#, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Sensor1')
        ax.set_ylabel('Sensor2')
        plt.show()
        # Correlation 1+2+3+4
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j] == 3.0:
                        color = 'blue'
                    elif ScaffoldingMatrix[i][j] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'white':
                        ax.scatter(i, j, c=color, linewidth=0)#, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Sensor1')
        ax.set_ylabel('Sensor2')
        plt.show()
        # Correlation 1+2+3+4 same color
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(ScaffoldingMatrix)):  # range(number):
            for j in range(len(ScaffoldingMatrix)):
                    if ScaffoldingMatrix[i][j] == 0.0:
                        color = 'red'
                    elif ScaffoldingMatrix[i][j] == 1.0:
                        color = 'green'
                    elif ScaffoldingMatrix[i][j] == 2.0:
                        color = 'orange'
                    elif ScaffoldingMatrix[i][j] == 3.0:
                        color = 'cyan'
                    elif ScaffoldingMatrix[i][j] == 4.0:
                        color = 'cyan'
                    else:
                        color = 'white'
                    # Dibujo el punto
                    if color != 'red' and color != 'white':
                        ax.scatter(i, j, c=color, linewidth=0)#, marker='s', s=60, linewidth=0.5)  # para mapa continuo sin s y sin linewidth
        ax.set_xlabel('Sensor1')
        ax.set_ylabel('Sensor2')
        plt.show()

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
            ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5,
                    marker='.', markersize=fi)
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

                # cn = colors.Normalize(min(active_corr), max(active_corr))  # creates a Normalize object for these z values
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
                    # ax.plot(sens1[i:i + 2], sens2[i:i + 2], sens3[i:i + 2], color=plt.cm.jet(cn(active_corr[i])), linewidth=0.5, marker='.', markersize=fi)
                    ax.plot(sens1[k:k + 2] + sens1[k + 1:k + 3] + sens1[k - 1:k + 1],
                            sens2[k:k + 2] + sens2[k + 1:k + 3] + sens2[k - 1:k + 1],
                            sens3[k:k + 2] + sens3[k + 1:k + 3] + sens3[k - 1:k + 1], color=plt.cm.jet(coef_deg),
                            linewidth=0.0,
                            marker='.', markersize=markers)
                    # ax.plot(sens1[k:k + 2], sens2[k:k + 2], sens3[k:k + 2], color=color, linewidth=0.5, marker='.', markersize=fi)

                    plt.show()

    def saveData(self):  # , action, seed):
        f = open('SimulationDataImplicitSensorizationPruebaVFCorregido_TrazasMUestraVF3060.pckl', 'wb')
        pickle.dump(len(self.correlationsManager.correlations), f)
        for i in range(len(self.correlationsManager.correlations)):
            pickle.dump(self.correlationsManager.correlations[i].S1_pos, f)
            pickle.dump(self.correlationsManager.correlations[i].S1_neg, f)
            pickle.dump(self.correlationsManager.correlations[i].S2_pos, f)
            pickle.dump(self.correlationsManager.correlations[i].S2_neg, f)
            # pickle.dump(self.correlationsManager.correlations[i].S3_pos, f)
            # pickle.dump(self.correlationsManager.correlations[i].S3_neg, f)
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
        f = open('SimulationDataImplicitSensorizationPruebaVFCorregido3.pckl', 'rb') #SimulationDataImplicitSensorizationPruebaIgBueno2 #SURS
                                                                                # SimulationDataImplicitSensorizationPruebaIgBueno # SURs trazas largas
                                                                                 #SimulationDataImplicitSensorizationPruebaVFCorregido #VF
                                                                                # SimulationDataImplicitSensorizationPruebaVFCorregido3
        numero = pickle.load(f)
        # for i in range(3):
        for i in range(numero):
            self.correlationsManager.correlations.append(Correlations(None))

        for i in range(numero):
            self.correlationsManager.correlations[i].S1_pos = pickle.load(f)
            self.correlationsManager.correlations[i].S1_neg = pickle.load(f)
            self.correlationsManager.correlations[i].S2_pos = pickle.load(f)
            self.correlationsManager.correlations[i].S2_neg = pickle.load(f)
            # self.correlationsManager.correlations[i].S3_pos = pickle.load(f)
            # self.correlationsManager.correlations[i].S3_neg = pickle.load(f)
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
        # # self.graph3 = pickle.load(f)
        # # self.graph4 = pickle.load(f)
        # # self.graph5 = pickle.load(f)
        # # self.graph6 = pickle.load(f)
        # # self.graph7 = pickle.load(f)
        self.TracesMemoryVF = pickle.load(f)
        f.close()

    def saveScaffoldingMatrix(self):
        '''Ad hoc for the paper'''
        Map1 = self.correlationsManager.correlations[0].S1_neg.getCertaintyMatrix()
        Map2 = self.correlationsManager.correlations[1].S2_neg.getCertaintyMatrix()
        Map3 = self.correlationsManager.correlations[2].S1_neg.getCertaintyMatrix()
        Map4 = self.correlationsManager.correlations[3].S1_neg.getCertaintyMatrix()
        # Create Scaffolding matrix
        ScaffoldingMatrix = Map1
        for x in range(len(Map2)):
            for y in range(len(Map2)):
                    if ScaffoldingMatrix[x][y] == 0.0:
                        if Map2[x][y]:
                            ScaffoldingMatrix[x][y] = 2
                        elif Map3[x][y]:
                            ScaffoldingMatrix[x][y] = 3
                        elif Map4[x][y]:
                            ScaffoldingMatrix[x][y] = 4
        # print ScaffoldingMatrix

        f = open('scaffoldingMatrixSURs.pckl', 'wb')
        pickle.dump(ScaffoldingMatrix, f)
        f.close()

    def saveScenarioMatrix(self):
        scenario_matrix = []
        # row = []
        for i in range(800, 40, -10):  # for i in range(1250,2400,5):
            row = []
            for j in range(1250, 2410, 10):  # for j in range(50,800,5):
                self.simulator.baxter_larm_set_pos((j, i))  # a.simulator.baxter_larm_set_pos((i,j))
                self.simulator.ball_set_pos(self.simulator.baxter_larm_act_get_pos())
                p = self.simulator.get_sensorization()
                active_corr = self.correlationsManager.getActiveCorrelationPrueba(p)
                corr_sensor, corr_type = self.correlationsManager.getActiveCorrelation(p, active_corr)
                if corr_sensor < 1:
                    row.append('')
                else:
                    row.append(active_corr)
            scenario_matrix.append(row)

        f = open('ScenarioMatrix.pckl', 'wb')
        pickle.dump(scenario_matrix, f)
        f.close()

    def saveScenarioMatrix2(self):
        scenario_matrix = []
        # row = []
        for i in range(800, 40, -10):  # for i in range(1250,2400,5):
            row = []
            for j in range(1250, 2410, 10):  # for j in range(50,800,5):
                if self.simulator.get_distance((j, i), self.simulator.balise1_get_pos()) < 20:
                    row.append(100)
                elif self.simulator.get_distance((j, i), self.simulator.balise2_get_pos()) < 20:
                    row.append(200)
                elif self.simulator.get_distance((j, i), self.simulator.box1_get_pos()) < 75:
                    row.append(300)
                    print 300
                else:
                    row.append(0)
            scenario_matrix.append(row)
        f = open('ScenarioMatrix2.pckl', 'wb')
        pickle.dump(scenario_matrix, f)
        f.close()

        print scenario_matrix

    def plotScenarioMatrix(self):
        f = open('ScenarioMatrix.pckl', 'rb')
        matrix = pickle.load(f)
        f.close()

        f = open('ScenarioMatrix2.pckl', 'rb')
        matrix2 = pickle.load(f)
        f.close()

        plt.figure()
        ax = plt.subplot(111)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix2[i][j] < 100:
                    if matrix[i][j] == 0.0:
                        color = 'red'
                    elif matrix[i][j] == 1.0:
                        color = 'green'
                    elif matrix[i][j] == 2.0:
                        color = 'orange'
                    elif matrix[i][j] == 3.0:
                        color = 'blue'
                    elif matrix[i][j] == 4.0:
                        color = 'cyan'
                    elif matrix[i][j] == 5.0:
                        color = 'pink'
                    elif matrix[i][j] == 6.0:
                        color = 'yellow'
                    elif matrix[i][j] == 7.0:
                        color = 'purple'
                    elif matrix[i][j] == 8.0:
                        color = 'white'
                    elif matrix[i][j] == 9.0:
                        color = 'olive'
                    elif matrix[i][j] == 10.0:
                        color = 'brown'
                    # elif matrix[i][j] == 11.0:
                    #     color = 'gray'
                    # else:
                    #     color = [0, 0, 0]
                    else:
                        color = 'gray'
                else:
                    if matrix2[i][j] == 100.0:
                        color = 'pink'
                    elif matrix2[i][j] == 200.0:
                        color = 'cyan'
                    elif matrix2[i][j] == 300.0:
                        color = [0,0,0]
                ax.scatter(j, i, c=color, marker='s', s=30, linewidth=0)

    def DrawTracesVF(self):
        plt.figure()
        # Traces VF
        plt.title('Traces VF', fontsize=15.0)
        for i in range(30):# for i in range(25,52):
            Trace = self.TracesMemoryVF.getTracesList()[-i]
            x = []
            y = []
            for i in range(len(Trace)):
                x.append(Trace[i][0][0])
                y.append(Trace[i][0][1])
            # plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
            plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
            plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
            col = [ i/30.0 , 1-i/30.0 , 0.5]
            plt.plot(x, y, marker='.', color=col, linewidth=2)  # 8
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
        # plt.xticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
        # plt.yticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
        plt.grid()

        # Traces SUR
        plt.figure()
        plt.title('Traces SUR', fontsize=15.0)
        for i in range(16,35):
            Trace = self.TracesMemoryVF.getTracesList()[i]
            x = []
            y = []
            for i in range(len(Trace)):
                x.append(Trace[i][0][0])
                y.append(Trace[i][0][1])
            # plt.plot(x, y, marker='.', color='orange', linewidth=2)  # 8
            plt.scatter(Trace[-1][0][0], Trace[-1][0][1], color='red', linewidth=4.5)
            plt.scatter(Trace[0][0][0], Trace[0][0][1], color='green', linewidth=4.5)
            plt.plot(x, y, marker='.', color='blue', linewidth=2)  # 8
        plt.xlabel('db1 (m)', fontsize=15.0)
        plt.ylabel('db2 (m)', fontsize=15.0)
        x=[]
        y=[]
        for i in range(0,1000,10):
            x.append(i)
            x.append(0)
            y.append(0)
            y.append(i)
            plt.plot(x,y,marker='.', color='grey')
            x=[]
            y=[]
        plt.xlim(0, 1200)
        plt.ylim(0, 1200)
        # plt.xticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
        # plt.yticks([0, 200, 400, 600, 800, 1000, 1200],['0', '0.2','0.4','0.6', '0.8','1.0', '1.2'])
        plt.grid()

def main():
    instance = MDBCore()
    instance.run()


if __name__ == '__main__':
    main()


# import pickle;f=open('store.pckl', 'rb');seed=pickle.load(f);f.close();import numpy as np;np.random.set_state(seed);from MDBCore import *;a=MDBCore()
            # a.run()
# seed=np.random.get_state();import pickle;f=open('seed_object.pckl', 'wb');pickle.dump(seed,f);f.close()

# Data=a.correlationsManager.correlations[0].S2_neg.TracesMemory.getTracesList()
# Data2=a.correlationsManager.correlations[0].S2_neg.TracesMemory.getWeakTracesList()
# fichero = open ( 'DataPruebaRed.txt', 'a' ) 
# for i in range(len(Data)):
#     ExpUtility = 1.0
#     for j in range(len(Data[i])):
#         fichero.write(str(Data[i][-1-j][0]) + ' ' + str(Data[i][-1-j][1]) + ' ' + str(Data[i][-1-j][2]) + ' ' + str(ExpUtility)+'\n')
#         ExpUtility -= ExpUtility/len(Data[i])
# for i in range(len(Data2)):
#     ExpUtility = 1.0
#     for j in range(len(Data2[i])):
#         fichero.write(str(Data2[i][-1-j][0])+ ' ' +str(Data2[i][-1-j][1])+ ' ' +str(Data2[i][-1-j][2])+ ' ' + str(ExpUtility)+'\n')
#         ExpUtility -= ExpUtility/len(Data2[i])
# fichero.close()

# Data=a.correlationsManager.correlations[0].S2_neg.TracesMemory.getTracesList()
# Data2=a.correlationsManager.correlations[0].S2_neg.TracesMemory.getWeakTracesList()
# fichero = open ( 'DataPruebaRedSensor2Norm.txt', 'a' )
# for i in range(len(Data)):
#     ExpUtility = 1.0
#     for j in range(len(Data[i])):
#         fichero.write(str(Data[i][-1-j][1]/1373.0)  + ' ' + str(ExpUtility)+'\n')
#         ExpUtility -= ExpUtility/len(Data[i])
# for i in range(len(Data2)):
#     ExpUtility = 1.0
#     for j in range(len(Data2[i])):
#         fichero.write(str(Data2[i][-1-j][1]/1373.0) + ' ' + str(ExpUtility)+'\n')
#         ExpUtility -= ExpUtility/len(Data2[i])
# fichero.close()


# Trazas IDEALES
# f=open('TrazasHAISPaperIdeal.pckl', 'rb')
# Trace1=pickle.load(f)
# Trace2=pickle.load(f)
# Trace3=pickle.load(f)
# f.close()
#
# plt.figure()
# # plt.title('Traces VF', fontsize=15.0)
# x = []
# y = []
# for i in range(len(Trace1)):
#     x.append(Trace1[i][0])
#     y.append(Trace1[i][1])
# plt.scatter(Trace1[-1][0], Trace1[-1][1], color='red', linewidth=4.5)
# plt.scatter(Trace1[0][0], Trace1[0][1], color='green', linewidth=4.5)
# # col = [i / 30.0, 1 - i / 30.0, 0.5]
# plt.plot(x, y, marker='.', linewidth=2)  # 8
#
# x = []
# y = []
# for i in range(len(Trace2)):
#     x.append(Trace2[i][0])
#     y.append(Trace2[i][1])
# plt.scatter(Trace2[-1][0], Trace2[-1][1], color='red', linewidth=4.5)
# plt.scatter(Trace2[0][0], Trace2[0][1], color='green', linewidth=4.5)
# # col = [i / 30.0, 1 - i / 30.0, 0.5]
# plt.plot(x, y, marker='.', linewidth=2)  # 8
#
# x = []
# y = []
# for i in range(len(Trace3)):
#     x.append(Trace3[i][0])
#     y.append(Trace3[i][1])
# plt.scatter(Trace3[-1][0], Trace3[-1][1], color='red', linewidth=4.5)
# plt.scatter(Trace3[0][0], Trace3[0][1], color='green', linewidth=4.5)
# # col = [i / 30.0, 1 - i / 30.0, 0.5]
# plt.plot(x, y, marker='.', linewidth=2)  # 8
#
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
# # Draw beacons
# plt.scatter(1000, 0, marker='s', color='blue', linewidth=15)
# plt.scatter(0, 1000, marker='s', color='green', linewidth=15)
# plt.scatter(1000, 0, marker='s', color='blue', linewidth=20)
# plt.scatter(0, 1000, marker='s', color='green', linewidth=20)
