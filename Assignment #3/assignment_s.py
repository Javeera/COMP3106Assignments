# Fatema Lokhandwala (101259465)
# Gurleen Bassali (101260100)
# Javeera Faizi (101191910)

# import statements
import os
import csv
from collections import defaultdict

class td_qlearning:
  eps = 1e-6
  max_iterations = 1000
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
        # Read CSV using the built-in csv module (each row expected to be [state, action])
        trial_seq = [] # list of (state, action) tuples for this trial
        with open(filepath, newline='') as csvfile:
          reader = csv.reader(csvfile)
          for row in reader:
            if len(row) < 2:
              continue
            s, a = row[0], row[1]

            state = str(s).strip()
            action = None if a.strip() == '-' else int(a)
            reward = self.reward(state)
            trial_seq.append((state, action)) # add (state, action) pair to sequence

            # don't update Q-value for terminal states
            if action is not None:
              self.Q[(state, action)] = reward # initially estimate Q(s,a) = r(s) for all state-action pairs observed in the trials
        self.trials.append(trial_seq) # add trial sequence to trials
        #print(self.Q)
        #print(self.rewards)
    # Run Q-learning until convergence
    # Logic:
    # Each adjacent pair of rows in the csv is a transition (s, a) -> s'
    # For every state transition in all trials:
      # Update Q(s,a) using the Q-learning update equation:
        # Computes the target for the state-action pair
        # Moves Q(s,a) towards target
      # Also track the maximum change in Q-values during the iteration
    # After processing all trials, check if the maximum change is below a small threshold (eps)
      # If so, we consider the Q-values to have converged and stop iterating
    # Repeat over all trials until values stop changing (converge).

    # Iterate over each sequence of trials
    for i in range(self.max_iterations):
      change = 0 # track maximum change in Q-values for convergence check
      for trial_seq in self.trials: 
        for k in range(len(trial_seq) - 1): # check each adjacent pair of (s, a) in the sequence
          state, action = trial_seq[k] 
          state_next, action_next = trial_seq[k-1]

          # if terminal state --> can't update Q-value
          if action is None: 
              continue
          action = int(action) # convert action to integer ###########s
          
          #print(f"Updating state={state}, action={action}, next={state_next}")  # Debugging print statement

          old = self.Q.get((state, action), self.reward(state)) # current Q-value
          new = self.update(state, action, state_next)          # updated Q-value
          change = max(change, abs(new - old))                  # maximum change in Q-values

      # Check for convergence: iterations barely change the numbers anymore --> Q-table has stabilized
      if change < self.eps:
        break

  # Qvalue function
  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action
    Q_value = self.Q.get((state, action), self.reward(state))
    if Q_value is None:
        # Fallback Q-value Q(s,a)=r(s)
        Q_value = self.reward(state)

    # Return the learned q-value for the state-action pair
    return round(Q_value, 2)
    #return Q_value

  # Policy function - returns the optimal action for a given state
  def policy(self, state):
    # state is a string representation of a state
    actions = self.available_actions(state)
    if not actions:
      return 0  # no available actions, return 0

    scores = [] # list of (action, Q value) tuples
    base_reward = self.reward(state) 
    for action in actions:
      Q_value = self.Q.get((state, action), base_reward) # get Q value or use base reward if not found
      scores.append((action, Q_value)) # store all actions with their Q values
    best_q = max(q for a, q in scores) # find the best Q value

    # for a, q in scores:
    #   if q == best_q: # tie --> return any optimal action
    #     return a      # return the first action with the best q value
    best_actions = [a for a, q in scores if q == best_q]
    return max(best_actions)

  # HELPERS

  # Return the reward for a given state
  def reward(self, state):
    # if state in self.rewards: #i.e. if we already calculated the reward for this state
    #   print("NO WINNER YET") #debugging print
    #   return self.rewards[state]
    try:
      c_bag = int(state.split('/')[0])
      c_agent = int(state.split('/')[1])
      c_opponent = int(state.split('/')[2])
      winner = state.split('/')[3]
    except:
      return 0
    #print("current state", state)
    if winner == 'A':   # agent wins
      #print("AGENT WINS!!!", c_agent) #debugging print
      reward = c_agent  # return reward
    elif winner == 'O': # opponent wins
      #print("OPPONENT WINS!!!", -c_agent) #debugging print
      reward = -c_agent # return negative reward
    else:               # non terminal state
      reward = 0        # no reward
    
    self.rewards[state] = reward
    #print("REWARD for state", state, "is", reward) #debugging print
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
    if r_next != 0:
      #print(r_next," + ", self.gamma,"*",max_next) #debugging print
      print("")
    return r_next + self.gamma * max_next

  # Update function for the Q-learning update
  def update(self, state, action, state_next):
    # state is the string representation of the current state
    # action is the integer representation of the action taken
    # state_next is the string representation of the next state
    # Q-learning update:
    #   Q(s,a) ← Q(s,a) + α * (r(s') + γ max_{a'} Q(s',a') - Q(s,a))
    # Returns the new Q(s,a).
    
    # Q-learning update rule
    current_q = self.Q.get((state, action), self.reward(state)) # default to reward if Q value not found
    # r_s = self.reward(state_next)
      
    # # if next state is terminal (ends with /A or /O), assign its reward to current state
    # #if state_next.endswith('/A') or state_next.endswith('/O'):
    # if r_s != 0:
    #   print("TERMINAL STATE!!!! R_S:", r_s)
    #   #max(self.Q.get((s_next, a_p), r_next) for a_p in actions_next)
    #   target_next = r_s
    # else:
    #   #actions_next = self.available_actions(state_next)
    # target_next = self.target(state_next) #r_s + self.gamma * max_next #computing next target state
    target_next = r_s + self.gamma * max_next #computing next target state
    # if target_next == 7:
    #   #print("TARGET NEXT:", target_next)
    #   print("")

    q_new = current_q + self.alpha * (target_next - current_q)
    self.Q[(state, action)] = q_new                             # update Q table estimate
    #print("Q_NEW:", q_new)
    return q_new

#ex 0
dir_path = "Examples/Example0/Trials"
agent = td_qlearning(dir_path)

print("State: 8/3/2/-")
print(" Q-value:", agent.qvalue("8/3/2/-", 2), " | Expected Q:", 5.67,
      "| Policy:", agent.policy("8/3/2/-"), " | Expected Policy:", 2)
print()

#ex 1
dir_path = "Examples/Example1/Trials"
agent = td_qlearning(dir_path)

print("State: 6/1/6/-")
print(" Q-value:", agent.qvalue("6/1/6/-", 2), " | Expected Q:", 0, "| Policy:", agent.policy("6/1/6/-"), " | Expected Policy:", 2)
print()

print("State: 0/7/6/O")
print(" Q-value:", agent.qvalue("0/7/6/O", 0), " | Expected Q:", -7, "| Policy:", agent.policy("0/7/6/O"), " | Expected Policy: terminal / none")
print()

print("State: 1/8/4/-")
print(" Q-value:", agent.qvalue("1/8/4/-", 1), " | Expected Q:", -8.10, "| Policy:", agent.policy("1/8/4/-"), " | Expected Policy:", 1)
print()

print("State: 1/6/6/-")
print(" Q-value:", agent.qvalue("1/6/6/-", 1), " | Expected Q:", 0, "| Policy:", agent.policy("1/6/6/-"), " | Expected Policy:", 1)
print()

print("State: 4/5/4/-")
print(" Q-value:", agent.qvalue("4/5/4/-", 3), " | Expected Q:", 7.20, "| Policy:", agent.policy("4/5/4/-"), " | Expected Policy:", 3)
print()

print("State: 2/8/3/-")
print(" Q-value:", agent.qvalue("2/8/3/-", 2), " | Expected Q:", -9.00, "| Policy:", agent.policy("2/8/3/-"), " | Expected Policy:", 2)
print()

print("State: 9/2/2/-")
print(" Q-value:", agent.qvalue("9/2/2/-", 3), " | Expected Q:", 5.83, "| Policy:", agent.policy("9/2/2/-"), " | Expected Policy:", 3)
print()

print("State: 3/6/4/-")
print(" Q-value:", agent.qvalue("3/6/4/-", 2), " | Expected Q:", 7.20, "| Policy:", agent.policy("3/6/4/-"), " | Expected Policy:", 2)
print()

print("State: 1/5/7/-")
print(" Q-value:", agent.qvalue("1/5/7/-", 1), " | Expected Q:", 0, "| Policy:", agent.policy("1/5/7/-"), " | Expected Policy:", 1)
print()

print("State: 1/9/3/-")
print(" Q-value:", agent.qvalue("1/9/3/-", 1), " | Expected Q:", -9.00, "| Policy:", agent.policy("1/9/3/-"), " | Expected Policy:", 1)
print()


#ex 2
dir_path = "Examples/Example2/Trials"
agent = td_qlearning(dir_path)

print("State: 8/3/2/-")
print(" Q-value:", agent.qvalue("8/3/2/-", 3), " | Expected Q:", 7.29, "| Policy:", agent.policy("8/3/2/-"), " | Expected Policy:", 3)
print()

print("State: 1/8/4/-")
print(" Q-value:", agent.qvalue("1/8/4/-", 1), " | Expected Q:", -8.10, "| Policy:", agent.policy("1/8/4/-"), " | Expected Policy:", 1)
print()

print("State: 2/8/3/-")
print(" Q-value:", agent.qvalue("2/8/3/-", 2), " | Expected Q:", 0, "| Policy:", agent.policy("2/8/3/-"), " | Expected Policy:", 2)
print()

print("State: 13/0/0/-")
print(" Q-value:", agent.qvalue("13/0/0/-", 1), " | Expected Q:", -2.6244, "| Policy:", agent.policy("13/0/0/-"), " | Expected Policy:", 1)
print()

print("State: 4/4/5/-")
print(" Q-value:", agent.qvalue("4/4/5/-", 1), " | Expected Q:", 0, "| Policy:", agent.policy("4/4/5/-"), " | Expected Policy:", 1)
print()

print("State: 8/2/3/-")
print(" Q-value:", agent.qvalue("8/2/3/-", 3), " | Expected Q:", 0, "| Policy:", agent.policy("8/2/3/-"), " | Expected Policy:", 3)
print()

print("State: 11/0/2/-")
print(" Q-value:", agent.qvalue("11/0/2/-", 2), " | Expected Q:", 0, "| Policy:", agent.policy("11/0/2/-"), " | Expected Policy:", 2)
print()

print("State: 0/3/10/O")
print(" Q-value:", agent.qvalue("0/3/10/O", 0), " | Expected Q:", -3, "| Policy:", agent.policy("0/3/10/O"), " | Expected Policy: terminal / none")
print()

print("State: 10/0/3/-")
print(" Q-value:", agent.qvalue("10/0/3/-", 3), " | Expected Q:", 0, "| Policy:", agent.policy("10/0/3/-"), " | Expected Policy:", 3)
print()

print("State: 5/5/3/-")
print(" Q-value:", agent.qvalue("5/5/3/-", 1), " | Expected Q:", -5.67, "| Policy:", agent.policy("5/5/3/-"), " | Expected Policy:", 1)
print()


#ex 3
dir_path = "Examples/Example3/Trials"
agent = td_qlearning(dir_path)

print("State: 1/7/5/-")
print(" Q-value:", agent.qvalue("1/7/5/-", 1), " | Expected Q:", -7.20, "| Policy:", agent.policy("1/7/5/-"), " | Expected Policy:", 1)
print()

print("State: 9/1/3/-")
print(" Q-value:", agent.qvalue("9/1/3/-", 3), " | Expected Q:", -4.37, "| Policy:", agent.policy("9/1/3/-"), " | Expected Policy:", 3)
print()

print("State: 4/4/5/-")
print(" Q-value:", agent.qvalue("4/4/5/-", 2), " | Expected Q:", 0, "| Policy:", agent.policy("4/4/5/-"), " | Expected Policy:", 2)
print()

print("State: 4/3/6/-")
print(" Q-value:", agent.qvalue("4/3/6/-", 1), " | Expected Q:", 0, "| Policy:", agent.policy("4/3/6/-"), " | Expected Policy:", 1)
print()

print("State: 6/3/4/-")
print(" Q-value:", agent.qvalue("6/3/4/-", 3), " | Expected Q:", 0, "| Policy:", agent.policy("6/3/4/-"), " | Expected Policy:", 3)
print()

print("State: 2/4/7/-")
print(" Q-value:", agent.qvalue("2/4/7/-", 1), " | Expected Q:", 0, "| Policy:", agent.policy("2/4/7/-"), " | Expected Policy:", 1)
print()

print("State: 1/7/5/- (duplicate check)")
print(" Q-value:", agent.qvalue("1/7/5/-", 1), " | Expected Q:", -7.20, "| Policy:", agent.policy("1/7/5/-"), " | Expected Policy:", 1)
print()

print("State: 4/6/3/-")
print(" Q-value:", agent.qvalue("4/6/3/-", 2), " | Expected Q:", -7.29, "| Policy:", agent.policy("4/6/3/-"), " | Expected Policy:", 2)
print()

print("State: 13/0/0/-")
print(" Q-value:", agent.qvalue("13/0/0/-", 2), " | Expected Q:", -3.28, "| Policy:", agent.policy("13/0/0/-"), " | Expected Policy:", 2)
print()

print("State: 2/4/7/- (second action test)")
print(" Q-value:", agent.qvalue("2/4/7/-", 2), " | Expected Q:", 0, "| Policy:", agent.policy("2/4/7/-"), " | Expected Policy:", 2)
print()



#prev state and current state instead of next state.

# Print results
# 5.67

# 0
# -7
# -0.03
# 0
# -0.07
# 0.08
# -0.07
# -0.04
# 0
# 0.23

# -0.25
# -0.03
# 0.08
# -1.85
# 0
# 0
# 0
# -3
# 0
# -0.02

# -0.02
# 0
# 0
# 0
# 0
# 0
# -0.02
# -0.08
# 0.3
# 0