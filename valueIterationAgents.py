# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    # Algorithme d'itération par valeurs
    for _ in range(iterations):
            new_values = util.Counter()  # Stocke les valeurs pour cette itération
            for state in mdp.getStates():  # Parcourir tous les états du MDP
                possible_actions = mdp.getPossibleActions(state)
                if not possible_actions:  # Cas d'un état sans actions (état terminal)
                    new_values[state] = 0
                else:
                    # Calcul de la meilleure valeur possible pour cet état
                    new_values[state] = max(
                        sum(
                            prob * (mdp.getReward(state, action, nextState) +
                                    discount * self.values[nextState])
                            for nextState, prob in mdp.getTransitionStatesAndProbs(state, action)
                        )
                        for action in possible_actions
                    )
            self.values = new_values  # Mettre à jour les valeurs après l'itération
    '***fin de mon code***'
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    return sum(
            prob * (self.mdp.getReward(state, action, nextState) +
                    self.discount * self.values[nextState])
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )
   #"*** fin de mon code ***"
    

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    possible_actions = self.mdp.getPossibleActions(state)
    if not possible_actions:  # Aucun mouvement possible (état terminal)
      return None
    return max(
            possible_actions,
            key=lambda action: self.getQValue(state, action)  # Choisir l'action avec la meilleure Q-valeur
        )
    #fin code
    

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
