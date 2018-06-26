class GoalManager(object):
    """Class that represents the Goal Manager
    
    This component includes different processes involved in achieving the goal,
    mainly the sub-goal identification and combination. It provides the Candidate
    State Evaluator with some criteria to evaluate the sensorial candidate states,
    and it requires the current set of motivations from the Motivation Manager in
    order to create the sub-goals.
    
    """

    def __init__(self):



    def goalAchieved(self, reward):
        if reward:
            return 1
        else:
            return 0
