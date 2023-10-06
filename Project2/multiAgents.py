# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        if "Stop" in legalMoves:
            stop_index = legalMoves.index("Stop")
            legalMoves.pop(stop_index)

        # Choose one of the best actions
        scores = [
            self.evaluationFunction(gameState, action) for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(
            bestIndices
        )  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(
        self,
        currentGameState,
        action,
    ):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # Variables to compute the build a reward mechanism (kind of) for
        # defining the next state of pacman. Here we define three variables
        # which are for location of the closest food pellet when taking that
        # particular action (minimum distance to food), what is the distance
        # from the ghost and is the ghost within 1 location distance from the
        # pacman.
        (
            minimum_distance_to_ghost,
            minimum_distance_to_food,
            ghost_closeness,
        ) = (1, -1, 0)

        # Computing the distance of the agent to the ghost and making sure that
        # the agent does not hit the ghosts using ghost closeness metric.
        for ghost_location in successorGameState.getGhostPositions():
            computed_distance = util.manhattanDistance(newPos, ghost_location)
            minimum_distance_to_ghost += computed_distance
            if computed_distance <= 1:
                ghost_closeness = 1

        # Computing the distance of the agent to the food pellet.
        for food in newFood.asList():
            distances = util.manhattanDistance(newPos, food)
            if (
                minimum_distance_to_food >= distances
                or minimum_distance_to_food == -1
            ):
                minimum_distance_to_food = distances

        # Because we want the reward for the closest food to be higher, we
        # compute the fraction of the values, as max distance would be higher
        # numerically than minimum distances.
        return (
            successorGameState.getScore()
            + (1 / minimum_distance_to_food)
            - (1 / minimum_distance_to_ghost)
            - ghost_closeness
        )


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def compute_min(game_state, agent_index, depth):
            # Obtain the necessary parameters for the computation of the min
            # function.
            agent_count = game_state.getNumAgents()
            successor_actions = game_state.getLegalActions(agent_index)

            # If there are no legal actions in the given state, return the
            # computed evaluation score to generate the output for the game.
            if not successor_actions:
                return self.evaluationFunction(game_state)

            # Compute the maximum value of the selected agent is the last one
            # in the sequence and find the minimum of the scores computed for
            # all possible actions presented at that state.
            if agent_index == agent_count - 1:
                max_values = list()
                for action in successor_actions:
                    score = compute_max(
                        game_state=game_state.generateSuccessor(
                            agent_index, action
                        ),
                        depth=depth,
                    )
                    max_values.append(score)
                minimum_value = min(max_values)

            # Compute the minimum value of the selected agent is not the last
            # one in the sequence and find the minimum of the scores computed
            # for all possible actions presented at that state.
            else:
                min_values = list()
                for action in successor_actions:
                    score = compute_min(
                        game_state=game_state.generateSuccessor(
                            agent_index, action
                        ),
                        agent_index=agent_index + 1,
                        depth=depth,
                    )
                    min_values.append(score)

                minimum_value = min(min_values)

            return minimum_value

        def compute_max(game_state, depth):
            # Start the computaton the max function with the first available
            # agent and obtain the set of legal actions for the agent's state.
            agent_index = 0

            successor_actions = game_state.getLegalActions(agent_index)

            # If there are no available legal actions or if the maximum depth
            # is reached, return the computed evaluation score to generate the
            # output for the game.
            if not successor_actions or depth == self.depth:
                return self.evaluationFunction(game_state)

            # Compute the maximum value of the selected agent in the sequence
            # and find the maximum of the scores computed for all possible
            # actions presented at that state.
            maximum_values = list()
            for action in successor_actions:
                score = compute_min(
                    game_state=game_state.generateSuccessor(0, action),
                    agent_index=agent_index + 1,
                    depth=depth + 1,
                )
                maximum_values.append(score)

            return max(maximum_values)

        actions = gameState.getLegalActions(0)
        all_actions = dict()

        for action in actions:
            all_actions[action] = compute_min(
                game_state=gameState.generateSuccessor(0, action),
                agent_index=1,
                depth=1,
            )

        return max(all_actions, key=all_actions.get)  # type: ignore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
