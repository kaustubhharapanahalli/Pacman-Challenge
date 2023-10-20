# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import collections

import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        for _ in range(self.iterations):
            iteration_values = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    iteration_values[state] = 0

                else:
                    maximum_value = -99999
                    actions = self.mdp.getPossibleActions(state)

                    for action in actions:
                        transition = self.mdp.getTransitionStatesAndProbs(
                            state, action
                        )
                        value = 0

                        for state_and_prob in transition:
                            value += state_and_prob[1] * (
                                self.mdp.getReward(
                                    state, action, state_and_prob[1]
                                )
                                + (
                                    self.discount
                                    * self.values[state_and_prob[0]]
                                )
                            )

                        maximum_value = max(value, maximum_value)

                    if maximum_value != -99999:
                        iteration_values[state] = maximum_value

            self.values = iteration_values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        q_value = 0
        for state_and_prob in self.mdp.getTransitionStatesAndProbs(
            state, action
        ):
            q_value += state_and_prob[1] * (
                self.mdp.getReward(state, action, state_and_prob[1])
                + (self.discount * self.values[state_and_prob[0]])
            )

        return q_value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        all_actions = dict()

        for action in actions:
            all_actions[action] = self.computeQValueFromValues(
                state=state, action=action
            )

        return max(all_actions, key=all_actions.get)  # type: ignore

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        for iteration in range(self.iterations):
            state = states[iteration % len(states)]

            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)

                all_values = list()
                for action in actions:
                    q_value = self.getQValue(state=state, action=action)
                    all_values.append(q_value)

                self.values[state] = max(all_values)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        priority_queue = util.PriorityQueue()

        predecessor_states = dict()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for state_and_prob in self.mdp.getTransitionStatesAndProbs(
                        state, action
                    ):
                        if state_and_prob[0] in predecessor_states:
                            predecessor_states[state_and_prob[0]].add(state)
                        else:
                            predecessor_states[state_and_prob[0]] = {state}

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                q_values = list()
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    q_values.append(q_value)

                max_q_value = max(q_values)

                difference = abs(self.values[state] - max_q_value)
                priority_queue.update(state, -difference)

        for _ in range(self.iterations):
            if priority_queue.isEmpty():
                break

            state = priority_queue.pop()

            if not self.mdp.isTerminal(state):
                q_values = list()

                for action in self.mdp.getPossibleActions(state):
                    q_values.append(
                        self.computeQValueFromValues(state, action)
                    )

                self.values[state] = max(q_values)

            for predecessor_state in predecessor_states[state]:
                if not self.mdp.isTerminal(predecessor_state):
                    q_values = list()

                    for action in self.mdp.getPossibleActions(
                        predecessor_state
                    ):
                        q_values.append(
                            self.computeQValueFromValues(
                                predecessor_state, action
                            )
                        )

                    difference = abs(
                        self.values[predecessor_state] - max(q_values)
                    )

                    if difference > self.theta:
                        priority_queue.update(predecessor_state, -difference)
