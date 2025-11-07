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
    # list of trials, each trial is a list of (state, action) tuples
    self.trials = [] # [[(state, action), ...], [...], ...]

    #load all trials from the directory
    for file in os.listdir(directory):
      if file.endswith(".csv"):
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath, header=None)

      #gets all states and actions and initializes the q table with rewards
      seq = [] # list of (state, action) tuples for this trial
      for s, a in df.values:
        state = str(s).strip()
        action = str(a)
        reward = self.reward(state)
        seq.append((state, action)) # add (state, action) pair to sequence

        # don't update Q-value for terminal states
        if action != "-":
          self.Q[(state, action)] = reward # initially estimate Q(s,a) = r(s) for all state-action pairs observed in the trials
      self.trials.append(seq) # add trial sequence to trials

    print(self.Q)
    print(self.rewards)

    #TODO: Run Q-learning until convergence
    converged = False
    # Iterate over each sequence of trials
    for seq in self.trials:
      for i in range(len(seq) - 1):
        state, action = seq[i] 
        state_next, action_next = seq[i+1]

        # if terminal state --> can't update Q-value
        if action == "-": 
            continue
        action_next = int(action_next)          # convert action to integer
        self.update(state, action, state_next)  # update Q-value    

      # Each adjacent pair of rows in the csv is a transition (s, a) -> s'
      # For every state transition in all trials:
        # Update Q(s,a) using the Q-learning update equation
          # Computes the target for the state-action pair
          # Move Q(s,a) towards target
        # Repeat over all trials until values stop changing (converge).

  # Qvalue function
  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # Return the learned q-value for the state-action pair
    return round(self.Q.get((state, action)), 2)

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

  # HELPERS

  # Return the reward for a given state
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
  
  # Compute the target term for the next state for the Q-learning update equation
  def target(self, s_next):
    # s_next is the string representation of the next state

    """
      Compute the target term: r(s') + γ * max_{a'} Q(s', a')
      If s' is terminal (no actions), the max term is 0.
    """

    r_next = self.reward(s_next) # reward for next state
    actions_next = self.available_actions(s_next) # available actions in next state
    if not actions_next:
        return r_next  # terminal state --> no future term
    
    # compute max Q value for next state over all possible actions
    max_next = max(self.Q.get((s_next, a_p), r_next) for a_p in actions_next) # default to reward if Q value not found for (s', a')
    return r_next + self.gamma * max_next

  # Update function for the Q-learning update
  def update(self, state, action, state_next):
    # state is the string representation of the current state
    # action is the integer representation of the action taken
    # state_next is the string representation of the next state

    """
    Q-learning update:
      Q(s,a) ← Q(s,a) + α * (r(s') + γ max_{a'} Q(s',a') - Q(s,a))
    Returns the new Q(s,a).
    """

    # Q-learning update rule
    current_q = self.Q.get((state, action), self.reward(state)) # default to reward if Q value not found
    target = self.target(state_next)                            # compute target term
    q_new = current_q + self.alpha * (target - current_q)       # Q-learning update equation
    self.Q[(state, action)] = q_new                             # update Q table estimate
    return q_new


dir_path = "Examples/Example0/Trials"
agent = td_qlearning(dir_path)
