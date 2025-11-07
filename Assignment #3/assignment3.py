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
    # Return nothing
    self.q = {} #q table (state, action): value
    self.rewards = {} #state rewards

    #load all trials from the directory
    trails = []
    for file in os.listdir(directory):
      if file.endswith(".csv"):
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath, header=None)
        trails.append(df)

    #Q(s,a) = r(s)
    #gets the state and action and initializes the q table with rewards
    for trial in trails:
      for row in trial:
        if len(row) < 2:
          continue
        state = str(row[0]).strip()
        action = int(row[1])
        reward = self.rewards(state)
        self.q[(state,action)] = reward

    #TODO: Run Q-learning until convergence
    converged = False
      #go thru each trial (state, action)
      for trial in trials:
      #extract current state and action and next state
      
      #compute immediet rewards for next state
      #make sure actions r valid
      #if yes valid then update q value else use reward of next state
      #td


  def rewards(self, state):
    #return the reward for a given state
    if state in self.rewards: #i.e. if we already calculated the reward for this state
      return self.rewards[state]
    try:
      c_bag = int(state.split('/')[0])
      c_agent = int(state.split('/')[1])
      c_opponent = int(state.split('/')[2])
      winner = int(state.split('/')[3])
    except:
      return 0
    
    if winner == 'A':
      reward = c_agent #return reward
    elif winner == 'O':
      reward = -c_agent #return negative reward
    else:
      reward = 0
    
    self.rewards[state] = reward
    return reward
  
  def available_actions(self, state):
    # Return a list of integers representing the available actions in the given state
    try:
      c_bag = int(state.split('/')[0])
    except:
      return []

    if c_bag <= 0:
      return []
    return [i for i in [1, 2, 3] if i <= c_bag]

  # implementing qvalue function
  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # Return the q-value for the state-action pair
    return round(self.Q[state][action], 2)

  # implementing the policy function
  def policy(self, state):
    # state is a string representation of a state

    # Return the optimal action (as an integer) under the learned policy
    if not self.Q[state]:
      return 1
    else:
      return max(self.Q[state], key=self.Q[state].get)
