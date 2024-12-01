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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Initialize the evaluation score
        score = successorGameState.getScore()

        # -------- FOOD EVALUATION --------
        # Distance to the closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:  # If there is food
            closestFoodDistance = min(foodDistances)
            score += 10 / (closestFoodDistance + 1)  # Reward proximity to food

        # -------- FOOD COUNT PENALTY --------
        # Penalize for the total remaining food to encourage eating food quickly
        remainingFood = len(newFood.asList())
        score -= 4 * remainingFood
        
        # -------- GHOST EVALUATION --------
        # Penalize proximity to active ghosts
        ghostPenalty = 0
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDist = manhattanDistance(newPos, ghost.getPosition())
            if scaredTime > 0:  # Ghosts are scared
                score += 200 / (ghostDist + 1)  # Reward eating scared ghosts
            elif ghostDist < 2:  # Ghosts are dangerous
                ghostPenalty -= 500  # Heavily penalize if Pacman is too close to a ghost
        score += ghostPenalty

        return score
        
def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        bestScore = float("-inf")
        bestAction = None
        maxDepth = self.depth * gameState.getNumAgents()
        for action in actions:
          nextGameState = gameState.generateSuccessor(self.index,action)
          score = minimax(nextGameState, maxDepth, self.evaluationFunction, self.index)
          if score > bestScore: 
            bestScore = score
            bestAction = action
        return bestAction
        
def minimax(gameState, currentDepth, evalFun, agentIndex):
  if currentDepth == 1 or gameState.isWin() or gameState.isLose():
    return evalFun(gameState)
  
  totalAgents = gameState.getNumAgents()
  nextAgentIndex = (agentIndex + 1) % totalAgents
  
  actions = gameState.getLegalActions(nextAgentIndex)
  if nextAgentIndex == 0:
    maxEval = float("-inf")
    for action in actions:
      nextGameState = gameState.generateSuccessor(nextAgentIndex, action)
      maxEval = max(maxEval, minimax(nextGameState, currentDepth - 1, evalFun, nextAgentIndex))
    return maxEval
  else: 
    minEval = float("inf")
    for action in actions:
      nextGameState = gameState.generateSuccessor(nextAgentIndex, action)
      minEval = min(minEval, minimax(nextGameState, currentDepth - 1, evalFun, nextAgentIndex))
    return minEval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        maxDepth = self.depth * gameState.getNumAgents()
        
        alpha = float("-inf")
        beta = float("inf")
        for action in actions:
          nextGameState = gameState.generateSuccessor(self.index,action)
          score = alphabeta(nextGameState, maxDepth, self.evaluationFunction, self.index, alpha, beta)
          if score > alpha: 
            alpha = score
            bestAction = action
        return bestAction

def alphabeta(gameState, currentDepth, evalFun, agentIndex, alpha, beta):
  if currentDepth == 1 or gameState.isWin() or gameState.isLose():
    return evalFun(gameState)
  
  totalAgents = gameState.getNumAgents()
  nextAgentIndex = (agentIndex + 1) % totalAgents
  
  actions = gameState.getLegalActions(nextAgentIndex)
  if nextAgentIndex == 0:
    maxEval = float("-inf")
    for action in actions:
      nextGameState = gameState.generateSuccessor(nextAgentIndex, action)
      eval = alphabeta(nextGameState, currentDepth - 1, evalFun, nextAgentIndex, alpha, beta)
      maxEval = max(maxEval, eval)
      alpha = max(alpha, eval)
      if beta < alpha:
        break
    return maxEval
  else: 
    minEval = float("inf")
    for action in actions:
      nextGameState = gameState.generateSuccessor(nextAgentIndex, action)
      eval = alphabeta(nextGameState, currentDepth - 1, evalFun, nextAgentIndex, alpha, beta)
      minEval = min(minEval, eval)
      beta = min(beta, eval)
      if beta < alpha:
        break
    return minEval

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                return max(
                    expectimax(state.generateSuccessor(agentIndex, action), depth, 1)
                    for action in state.getLegalActions(agentIndex)
                )
            else:  # Ghost's turn (random choice)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                legalActions = state.getLegalActions(agentIndex)
                probabilities = 1 / len(legalActions) if legalActions else 0
                return sum(
                    probabilities
                    * expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                )

        legalMoves = gameState.getLegalActions(0)  # Pacman's legal actions
        scores = [expectimax(gameState.generateSuccessor(0, action), 0, 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index, score in enumerate(scores) if score == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Pacman's current position
    pacmanPos = currentGameState.getPacmanPosition()

    # Get food grid and convert to a list of positions
    foodList = currentGameState.getFood().asList()

    # Get ghost states and their scared timers
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Initialize evaluation score with the current game score
    score = currentGameState.getScore()

    # -------- FOOD EVALUATION --------
    # Reward for being closer to the nearest food
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    if foodDistances:
        closestFoodDistance = min(foodDistances)
        score += 10 / (closestFoodDistance + 1)  # Reward proximity to food

    # -------- FOOD COUNT PENALTY --------
    # Penalize remaining food to encourage Pacman to eat food faster
    remainingFood = len(foodList)
    if remainingFood <= 3:  # If there are very few food items left
        score += 100 / (remainingFood + 1)  # High reward for clearing the last few foods
    score -= 4 * remainingFood  # Penalize remaining food overall
    
    # -------- GHOST EVALUATION --------
    ghostPenalty = 0
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        ghostDist = manhattanDistance(pacmanPos, ghostState.getPosition())
        if scaredTime > 0:  # If the ghost is scared
            if scaredTime >= ghostDist:  # Pacman can reach the ghost in time
                score += 500 / (ghostDist + 1)  # Encourage Pacman to eat the ghost
            else:
                ghostPenalty -= 50 / (ghostDist + 1)  # Penalize moving toward unreachable scared ghost
        elif ghostDist <= 1:  # Strong penalty for ghosts within 1 step
            ghostPenalty -= 1000
        elif ghostDist <= 2:  # Moderate penalty for ghosts within 2 steps
            ghostPenalty -= 200 / ghostDist
    score += ghostPenalty

    # -------- EXPLORATION REWARD --------
    # Encourage Pacman to move to unexplored areas
    legalActions = currentGameState.getLegalActions()
    for action in legalActions:
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPacmanPos = successorGameState.getPacmanPosition()
        if newPacmanPos != pacmanPos:  # Reward movement to a new position
            score += 1  # Add a small reward for exploration
    
    return score

# Abbreviation
better = betterEvaluationFunction
