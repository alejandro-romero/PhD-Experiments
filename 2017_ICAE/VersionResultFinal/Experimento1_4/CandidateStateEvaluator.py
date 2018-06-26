# from MotivationManager import *
from ForwardModel import *
import numpy as np
from ActionChooser import *
from BackProp import *
import random

class CandidateStateEvaluator(object):
    def __init__(self):
        # self.MotivManager = MotivationManager()
        self.ForwModel = ForwardModel()
        self.actionChooser = ActionChooser()

        # Variables to control the Brownian motion (intrinsic motivation)
        self.n_random_steps = 0
        self.max_random_steps = 3

        self.intrinsic_exploration_type = 'Novelty'  # 'Brownian' or 'Novelty'

        self.n = 0.5  # Coefficient that regulates the balance between the relevance of distant and near states

    def getEvaluation(self, candidates, corr_sens, tipo, SimData, sensoriz_t):
        """Return the list os candidates actions sorted according to their value
        
        :param candidates: list o candidate actions
        :param corr_sens: number of the correlated sensor. 1 - sensor 1, 2 - sensor 2 ... n-sensor n
        :param tipo: type of the correlation: positive ('pos') or negative ('neg')
        :param SimData: data from the simulator needed to adjust the ForwardModel (baxter_pos, ball_pos, ball_situation, box_pos)
        :param sensoriz_t: actual sensorization to calculate the valuation
        :return: list of candidates actions with its valuation according to the active correlation
        """

        # type = type of the correlation: positive ('pos') or negative ('neg')
        evaluated_candidates = []
        for i in range(len(candidates)):
            valuation = self.getValuation(candidates[i], corr_sens, tipo, SimData, sensoriz_t)
            evaluated_candidates.append((candidates[i],) + (valuation,))

        # Ordenor los estados evaluados
        evaluated_candidates.sort(key=lambda x: x[-1])

        return evaluated_candidates

    def getValuation(self, candidate, sensor, tipo, SimData, sens_t):
        """Return the valuation for each individual candidate
        
        :param candidate: candidate action to evaluate
        :param sensor:  number of the correlated sensor. 1 - sensor 1, 2 - sensor 2 ... n-sensor n
        :param tipo: type of the correlation: positive ('pos') or negative ('neg')
        :param SimData: data from the simulator needed to adjust the ForwardModel (baxter_pos, ball_pos, ball_situation, box_pos)
        :param sens_t:  actual sensorization to calculate the valuation
        :return: valuation of the candidate state
        """
        # Obtengo valoracion aplicando la accion candidata en el modelo de mundo
        sens_t1 = self.ForwModel.predictedState(candidate, SimData)
        if tipo == 'pos':  # Tengo que alejarme, aumentar la distancia
            valuation = sens_t1[sensor - 1] - sens_t[sensor - 1]
        elif tipo == 'neg':  # Tengo que acercarme, disminuir la distancia
            valuation = sens_t[sensor - 1] - sens_t1[sensor - 1]

        return valuation

    # def getAction(self, explorationType, SimData, sensorialStateT1, corr_sensor, corr_type):
    #
    #     # explorationType = self.MotivManager.getActiveMotivation()
    #
    #     if explorationType == 'Int':  # Intrinsic Motivation
    #         # Brownian motion
    #         self.n_random_steps += 1
    #         if self.n_random_steps > self.max_random_steps:
    #             action = np.random.uniform(-45, 45)
    #             self.max_random_steps = np.random.randint(1, 4)
    #             self.n_random_steps = 0
    #         else:
    #             action = 0
    #     else:  # Extrinsic motivation -> Correlations
    #         candidate_actions = self.actionChooser.getCandidateActions()
    #         candidates_eval = self.getEvaluation(candidate_actions, corr_sensor, corr_type, SimData, sensorialStateT1)
    #         action = self.actionChooser.chooseAction(candidates_eval)
    #
    #     return action

    def getAction(self, explorationType, SimData, sensorialStateT1, corr_sensor, corr_type, intrinsicMemory, useVF, VFTracesMemory, trainNet):

        # explorationType = self.MotivManager.getActiveMotivation()

        if explorationType == 'Int':  # Intrinsic Motivation
            if self.intrinsic_exploration_type == 'Brownian':
                # Brownian motion
                self.n_random_steps += 1
                if self.n_random_steps > self.max_random_steps:
                    action = np.random.uniform(-45, 45)
                    self.max_random_steps = np.random.randint(1, 4)
                    self.n_random_steps = 0
                else:
                    action = 0
            elif self.intrinsic_exploration_type == 'Novelty':
                # action = 0
                candidate_actions = self.actionChooser.getCandidateActions()
                candidates_eval = self.getNoveltyEvaluation(candidate_actions, intrinsicMemory, SimData)
                action = self.actionChooser.chooseAction(candidates_eval)

        else:  # Extrinsic motivation -> Correlations
            candidate_actions = self.actionChooser.getCandidateActions()
            if useVF:  # Extrinsic motivation ->  VF
                candidates_eval = self.getVFEvaluation(candidate_actions,SimData, VFTracesMemory, trainNet)
            else:  # Extrinsic motivation -> SURs
                candidates_eval = self.getEvaluation(candidate_actions, corr_sensor, corr_type, SimData,
                                                     sensorialStateT1)
            action = self.actionChooser.chooseAction(candidates_eval)

        return action

    def getVFEvaluation(self, candidates, SimData, TracesListVF, trainNet):
        """Return the list os candidates actions sorted according to their VF value and following the active correlation

        :param candidates: list o candidate actions
        :param corr_sens: number of the correlated sensor. 1 - sensor 1, 2 - sensor 2 ... n-sensor n
        :param tipo: type of the correlation: positive ('pos') or negative ('neg')
        :param SimData: data from the simulator needed to adjust the ForwardModel (baxter_pos, ball_pos, ball_situation, box_pos)
        :param sensoriz_t: actual sensorization to calculate the valuation
        :param VFTraceMemory: memory with the last traces obtained to train the VF network
        :return: list of candidates actions with its valuation according to the VF value and the active correlation
        """

        evaluated_candidates = []
        valuations = self.getVFValuation(candidates, SimData, TracesListVF, trainNet)
        for i in range(len(candidates)):
            evaluated_candidates.append((candidates[i],) + (valuations[i],))
        # Ordenar los estados evaluados
        evaluated_candidates.sort(key=lambda x: x[-1])

        return evaluated_candidates

    def getVFValuation(self, candidates,SimData, TracesListVF, trainNet):
        """Return the VF valuation for each individual candidate

        :param candidate: candidate action to evaluate
        :param sensor:  number of the correlated sensor. 1 - sensor 1, 2 - sensor 2 ... n-sensor n
        :param tipo: type of the correlation: positive ('pos') or negative ('neg')
        :param SimData: data from the simulator needed to adjust the ForwardModel (baxter_pos, ball_pos, ball_situation, box_pos)
        :param sens_t:  actual sensorization to calculate the valuation
        :return: valuation of the candidate state
        """

        # Data to train the network
        train, test, valid, traintarget, testtarget, validtarget = self.getNormalisedData(TracesListVF)

        if trainNet:
            self.net = mlp(train, traintarget, 5, outtype='linear')
            self.net.mlptrain(train, traintarget, 0.25, 101)
            self.net.earlystopping(train, traintarget, valid, validtarget, 0.25)

        # Normalizo nueva sensorizacion, convierto en np.array y concateno el -1
        # Obtengo valoracion aplicando la accion candidata en el modelo de mundo
        valuations = []
        for i in range(len(candidates)):
            sens_t1 = self.ForwModel.predictedState(candidates[i], SimData)
            # Normalizo sens_t1
            sens_t1 = np.asarray(sens_t1+(-1,))
            sens_t1[0] /= (1300.0-0.0)  # Normalise
            sens_t1[1] /= (1300.0-0.0)  # Normalise
            sens_t1[2] /= (1300.0 - 0.0)  # Normalise
            valuations.append(self.net.mlpfwd(sens_t1.reshape(1, 4)))

        return valuations

    def getNormalisedData(self, TracesList):
        """Return normalised inputs and outputs from a list of Trace points

        :param TracesList: list o traces with the points used to train the net
        :return: arrays with normalised inputs and outputs shuffled and divided in training, validation and testing sets
        """
        # Network input and output
        in_data = []
        out_data = []
        # for i in range(len(TracesList)):
        #     for j in range(len(TracesList[i])):
        #         in_data.append(TracesList[i][j][0])
        #         out_data.append(TracesList[i][j][-1])
        for i in range(30, 60):
            for j in range(len(TracesList[-i])):
                in_data.append(TracesList[-i][j][0])
                out_data.append(TracesList[-i][j][-1])
        # Normalise inputs
        in_data = np.asarray(in_data)

        # Trabajo en el intervalo 0-1300 (maximo valor sensor)
        in_data /= 1300  # Divido entre valor maximo para normalizar entre 0 y 1

        # Data vector
        data = []
        for i in range(len(in_data)):
            data.append((in_data[i][0], in_data[i][1], in_data[i][2], out_data[i]))
        data = np.asarray(data)
        # Mix data to train the network
        random.shuffle(data)
        input = data[:, 0:2 + 1]
        output = data[:, 2 + 1]
        input = input.reshape((input.shape[0], 2 + 1))
        output = output.reshape((input.shape[0], 1))
        train = input[0::2, :]
        test = input[1::4, :]
        valid = input[3::4, :]
        traintarget = output[0::2, :]
        testtarget = output[1::4, :]
        validtarget = output[3::4, :]

        return train, test, valid, traintarget, testtarget, validtarget

    def getNoveltyEvaluation(self, candidates, trajectoryBuffer, SimData):
        """Return the list of candidates actions sorted according to their novelty value

        :param candidates: list o candidate actions
        :param trajectoryBuffer: buffer that stores the last perceptual states the robot has experienced
        :return: list of candidates actions sorted according to its novelty valuation
        """

        evaluated_candidates = []
        for i in range(len(candidates)):
            valuation = self.getNovelty(candidates[i], trajectoryBuffer, SimData)
            evaluated_candidates.append((candidates[i],) + (valuation,))

        # Ordenor los estados evaluados
        evaluated_candidates.sort(key=lambda x: x[-1])

        return evaluated_candidates

    def getNovelty(self, candidate_action, trajectoryBuffer, SimData):
        """Return the novelty for each individual candidate

        :param candidate: candidate action to evaluate its novelty
        :param trajectoryBuffer: buffer that stores the last perceptual states the robot has experienced
        :return: novelty of the candidate state
        """

        candidate_state = self.ForwModel.predictedState(candidate_action, SimData)
        novelty = 0
        for i in range(len(trajectoryBuffer)):
            novelty += pow(self.getDistance(candidate_state, trajectoryBuffer[i]), self.n)

        novelty = novelty / len(trajectoryBuffer)

        return novelty

    # def getDistance(self, (x1, y1), (x2, y2)):
    #     """Return the distance between two points"""
    #     return math.sqrt(pow(x2 - x1, 2) + (pow(y2 - y1, 2)))

    def getDistance(self, (x1, y1, z1), (x2, y2, z2)):
        """Return the distance between two points"""
        return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2))