# Fatema Lokhandwala (101259465)
# Gurleen Bassali (101260100)
# Javeera Faizi (101191910)

# import statements
import os
import csv
from collections import defaultdict
import pandas as pd

class td_qlearning:

  alpha = 0.10
  gamma = 0.90

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space
    # Constructor --> Returns nothing
    self.Q = {} #q table (state, action): value
    self.rewards = {} #state rewards

    #load all trials from the directory
    trials = []
    for file in os.listdir(directory):
      if file.endswith(".csv"):
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath, header=None)
        trials.append(df)

    #gets all states and actions and initializes the q table with rewards
    for trial in trials:
      for s, a in trial.values:
        state = str(s).strip()
        action = str(a)
        reward = self.reward(state)

        # don't update Q-value for terminal states
        if action != "-":
          self.Q[(state, action)] = reward # initially estimate Q(s,a) = r(s) for all state-action pairs observed in the trials

    print(self.Q)
    print(self.rewards)

    #TODO: Run Q-learning until convergence
    converged = False
      #go thru each trial (state, action)
      #for trial in trials:
      #extract current state and action and next state
      
      #compute immediet rewards for next state
      #make sure actions r valid
      #if yes valid then update q value else use reward of next state
      #td

  #return the reward for a given state
  def reward(self, state):
    if state in self.rewards: #i.e. if we already calculated the reward for this state
      return self.rewards[state]
    try:
      c_bag = int(state.split('/')[0])
      c_agent = int(state.split('/')[1])
      c_opponent = int(state.split('/')[2])
      winner = state.split('/')[3]
    except:
      return 0
    
    if winner == 'A':   # agent wins
      reward = c_agent  # return reward
    elif winner == 'O': # opponent wins
      reward = -c_agent # return negative reward
    else:               # non terminal state
      reward = 0        # no reward
    
    self.rewards[state] = reward
    return reward
  
  # Return a list of integers representing the available actions in the given state
  def available_actions(self, state):
    try:
      c_bag = int(state.split('/')[0]) # Remaining coins in the bag
    except:
      return []

    if c_bag <= 0:
      return []
    # Available actions are 1, 2, or 3 coins --> cannot exceed remaining coins in the bag
    return [i for i in [1, 2, 3] if i <= c_bag] 

  # Qvalue function
  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # Return the learned q-value for the state-action pair
    return round(self.Q[state][action], 2)

  # Policy function - returns the optimal action for a given state
  def policy(self, state):
    # state is a string representation of a state
    actions = self.available_actions(state)
    if not actions:
      return 1  # no available actions, return 1

    scores = [] # list of (action, Q value) tuples
    base_reward = self.reward(state) 
    for action in actions:
      Q_value = self.Q.get((state, action), base_reward) # get Q value or use base reward if not found
      scores.append((action, Q_value)) # store all actions with their Q values
    best_q = max(q for a, q in scores) # find the best Q value

    for a, q in scores:
      if q == best_q: # tie --> return any optimal action
        return a      # return the first action with the best q value

dir_path = "Examples/Example0/Trials"
agent = td_qlearning(dir_path)
