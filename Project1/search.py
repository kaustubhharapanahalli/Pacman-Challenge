# search.py
# ---------
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
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def genericSearch(problem, fringe):
    visited = set()
    totalPath = list()
    fringe.push((problem.getStartState(), list(), 0))
    while not fringe.isEmpty():
        currentState = fringe.pop()
        if problem.isGoalState(currentState[0]) == True:
            return currentState[1]
        if currentState[0] not in visited:
            for childNode, action, childCost in problem.getSuccessors(
                currentState[0]
            ):
                totalPath = currentState[1].copy()
                totalPath.append(action)
                totalCost = currentState[2] + childCost
                fringe.push((childNode, totalPath, totalCost))
        visited.add(currentState[0])

    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    dfs_stack = util.Stack()
    visited, path = list(), list()

    if problem.isGoalState(problem.getStartState()):
        return list()

    dfs_stack.push((problem.getStartState(), list()))

    while not dfs_stack.isEmpty():
        position, path = dfs_stack.pop()
        visited.append(position)

        if problem.isGoalState(position):
            return path

        successors = problem.getSuccessors(position)

        for successor in successors:
            if successor[0] not in visited:
                new_path = path + [successor[1]]
                dfs_stack.push((successor[0], new_path))

    return util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    bfs_queue = util.Queue()
    visited, path = list(), list()

    if problem.isGoalState(problem.getStartState()):
        return list()

    bfs_queue.push((problem.getStartState(), list()))

    while not bfs_queue.isEmpty():
        position, path = bfs_queue.pop()
        visited.append(position)

        if problem.isGoalState(position):
            return path

        successors = problem.getSuccessors(position)

        for successor in successors:
            if successor[0] not in visited and successor[0] not in (
                state[0] for state in bfs_queue.list
            ):
                new_path = path + [successor[1]]
                bfs_queue.push((successor[0], new_path))

    return util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    ucs_queue = util.PriorityQueue()
    visited, path = list(), list()

    if problem.isGoalState(problem.getStartState()):
        return list()

    ucs_queue.push((problem.getStartState(), list()), 0)

    while not ucs_queue.isEmpty():
        position, path = ucs_queue.pop()
        visited.append(position)

        if problem.isGoalState(position):
            return path

        successors = problem.getSuccessors(position)

        for successor in successors:
            if successor[0] not in visited and (
                successor[0] not in (node[2][0] for node in ucs_queue.heap)
            ):
                new_path = path + [successor[1]]
                previous = problem.getCostOfActions(new_path)

                ucs_queue.push((successor[0], new_path), previous)

            elif successor[0] not in visited and (
                successor[0] in (node[2][0] for node in ucs_queue.heap)
            ):
                for node in ucs_queue.heap:
                    if node[2][0] == successor[0]:
                        old_previous = problem.getCostOfActions(node[2][1])

                new_previous = problem.getCostOfActions(path + [successor[1]])

                if old_previous > new_previous:  # type: ignore
                    new_path = path + [successor[1]]
                    ucs_queue.update((successor[0], new_path), new_previous)

    return util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def f(problem, state, heuristic):
    return problem.getCostOfActions(state[1]) + heuristic(state[0], problem)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    astar_queue = util.PriorityQueue()
    visited, path = list(), list()

    if problem.isGoalState(problem.getStartState()):
        return list()

    state = (problem.getStartState(), list())

    cost = f(problem, state, heuristic)
    astar_queue.push((problem.getStartState(), list()), cost)

    while not astar_queue.isEmpty():
        states = astar_queue.pop()
        position, path = states

        if position in visited:
            continue

        visited.append(position)

        if problem.isGoalState(position):
            return path

        successors = problem.getSuccessors(position)

        for successor in successors:
            if successor[0] not in visited:
                new_path = path + [successor[1]]
                state = (successor[0], new_path)

                cost = f(problem, state, heuristic)
                astar_queue.push(state, cost)

    return util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
