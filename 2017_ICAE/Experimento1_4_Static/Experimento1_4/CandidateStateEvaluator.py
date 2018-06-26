# from MotivationManager import *
from ForwardModel import *
import numpy as np
from ActionChooser import *

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

    def getAction(self, explorationType, SimData, sensorialStateT1, corr_sensor, corr_type, intrinsicMemory):

        # explorationType = self.MotivManager.getActiveMotivation()

        if explorationType == 'Int':  # Intrinsic Motivation

            prob = .25
            self.intrinsic_exploration_type = np.random.choice(['Brownian', 'Novelty'], p=[prob, 1 - prob])

            if self.intrinsic_exploration_type == 'Brownian':
                # Brownian motion
                self.n_random_steps += 1
                if self.n_random_steps > self.max_random_steps:
                    action = (np.random.uniform(-90, 90), np.random.uniform(-90, 90))
                    self.max_random_steps = np.random.randint(1, 4)
                    self.n_random_steps = 0
                else:
                    action = (0, 0)
            elif self.intrinsic_exploration_type == 'Novelty':
                # action = 0
                self.n_random_steps = self.max_random_steps
                candidate_actions = self.actionChooser.getCandidateActions()
                candidates_eval = self.getNoveltyEvaluation(candidate_actions, intrinsicMemory, SimData)
                action = self.actionChooser.chooseAction(candidates_eval)
                # print "Action: ", action

        else:  # Extrinsic motivation -> Correlations
            candidate_actions = self.actionChooser.getCandidateActions()
            candidates_eval = self.getEvaluation(candidate_actions, corr_sensor, corr_type, SimData, sensorialStateT1)
            action = self.actionChooser.chooseAction(candidates_eval)
            # print "Action: ", action
        return action

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