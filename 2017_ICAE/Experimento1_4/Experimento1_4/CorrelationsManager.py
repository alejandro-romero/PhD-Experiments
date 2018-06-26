from Correlations import *
import logging


class CorrelationsManager(object):
    """Class that represents the Correlations Manager module.
        This module identifies when new correlations are needed and contains the set of existing correlations.

        It contains a list with all the existing Correlations.

        It also chooses the active correlation and gives the value of the reward based on this active correlation.
        """

    def __init__(self):

        self.correlations = []
        self.threshold = 0.1  # Threshold to know when to give reward to the sub-correlations

    def newCorrelation(self):
        """ This method decides when a new correlation has to be created
        
        :return: 
        """
        if len(self.correlations) == 0:
            self.correlations.append(Correlations(None))
            self.correlations[-1].figure.canvas.set_window_title('Correlation' + ' ' + str(len(self.correlations) - 1))
            logging.info('New correlation. Number of existing correlations: %s', len(self.correlations))

            self.correlations[-1].i_reward_assigned = 1

        if len(self.correlations) < 6:
            if self.correlations[-1].established:
                self.correlations.append(Correlations(len(self.correlations) - 1))
                #self.correlations.append(Correlations(0))
                self.correlations[-1].figure.canvas.set_window_title(
                    'Correlation' + ' ' + str(len(self.correlations) - 1))
                logging.info('New correlation. Number of existing correlations: %s', len(self.correlations))

    def getActiveCorrelation(self, p, active_corr):
        """ This method provides the active correlation among all the possible correlations for a given point p

        :return: active_correlation
        """
        # active_corr = len(self.correlations)-1
        # max_certainty = 0
        # for i in range(len(self.correlations)):
        #     certainty = self.correlations[i].getCertainty(p)
        #     if certainty > max_certainty:
        #         max_certainty = certainty
        #         active_corr = i

        corr_sensor, corr_type = self.correlations[active_corr].getActiveCorrelation(p)

        return corr_sensor, corr_type  # active_corr, corr_sensor, corr_type

    def getActiveCorrelationPrueba(self, p):

        active_corr = len(self.correlations)-1
        max_certainty = 0
        for i in range(len(self.correlations)):
            certainty = self.correlations[i].getCertainty(p)
            if certainty > max_certainty:
                max_certainty = certainty
                active_corr = i

        return active_corr

    # def getActiveCorrelation(self, p):
    #     """ This method provides the active correlation among all the possible correlations for a given point p
    #
    #     :return: active_correlation
    #     """
    #     active_corr = len(self.correlations)-1
    #     max_certainty = 0
    #     for i in range(len(self.correlations)):
    #         certainty = self.correlations[i].getCertainty(p)
    #         if certainty > max_certainty:
    #             max_certainty = certainty
    #             active_corr = i
    #
    #     corr_sensor, corr_type = self.correlations[active_corr].getActiveCorrelation(p)
    #
    #     return corr_sensor, corr_type, active_corr

    def getReward(self, active_corr, simulator, p):
        """This method is in charge of provide reward if required
        :param: active_corr: index of the active correlation needed to know who is providing its reward
        :return: reward
        """
        i_r = self.correlations[active_corr].i_reward

        # if i_r is None:
        #     reward = self.simulator.getReward()
        # elif self.correlations[i_r].getCertainty() > self.threshold:
        if i_r is None:
            reward = simulator
        elif self.correlations[i_r].getCertainty(p) > self.threshold:
            reward = 1
        else:
            reward = 0

        return reward

    def assignRewardAssigner(self, active_corr, p):

        for i in range(len(self.correlations[:active_corr])):
            if self.correlations[i].getCertainty(p) > self.threshold:
                self.correlations[active_corr].i_reward = i
                self.correlations[active_corr].i_reward_assigned = 1
                break

