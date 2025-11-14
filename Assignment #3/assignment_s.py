# Fatema Lokhandwala (101259465)
# Gurleen Bassali (101260100)
# Javeera Faizi (101191910)

# import statements
import os
import csv
from collections import defaultdict

class td_qlearning:
  eps = 0.00000001
  max_iterations = 1000
  alpha = 0.10
  gamma = 0.90

  def __init__(self, directory):
    self.Q = {}  # Q-table
    self.rewards = {}
    self.trials = []

    # Load all trials
    for file in os.listdir(directory):
      if file.endswith(".csv"):
        filepath = os.path.join(directory, file)
        trial_seq = []
        with open(filepath, newline='') as csvfile:
          reader = csv.reader(csvfile)
          for row in reader:
            if len(row) < 2:
              continue
            s, a = row[0], row[1]
            state = str(s).strip()
            action = None if a.strip() == '-' else int(a)
            trial_seq.append((state, action))
            # Initialize Q-values with reward (like Code 2)
            if action is not None:
              self.Q[(state, action)] = self.reward(state)
        self.trials.append(trial_seq)

    # Q-learning loop (like Code 2)
    for i in range(self.max_iterations):
      change = 0
      for trial_seq in self.trials:
        for k in range(len(trial_seq)):
          curr_state, curr_action = trial_seq[k]
          
          # Skip terminal states (like Code 2)
          if curr_action is None:
            continue
          
          curr_reward = self.reward(curr_state)  # Use current state's reward (like Code 2)
          old_q = self.Q.get((curr_state, curr_action), 0.0)
          
          # Check if next state exists
          if k + 1 < len(trial_seq):
            next_state, next_action = trial_seq[k + 1]
            
            # If next state is terminal, use its reward as next_q (like Code 2)
            if next_action is None:
              next_q = self.reward(next_state)
            else:
              # Get max Q-value for next state (like Code 2)
              actions = self.available_actions(next_state)
              max_q = float('-inf')
              for action in actions:
                q_value = self.Q.get((next_state, action), 0.0)
                max_q = max(max_q, q_value)
              next_q = max_q if max_q != float('-inf') else 0.0
          else:
            # Terminal state, no next state (like Code 2)
            next_q = 0.0

          # Q-learning update (like Code 2)
          new_q = old_q + self.alpha * (curr_reward + self.gamma * next_q - old_q)
          
          self.Q[(curr_state, curr_action)] = new_q
          change = max(change, abs(new_q - old_q))

      if change < self.eps:
        break

  # === Utility functions ===
  def qvalue(self, state, action):
    return self.Q.get((state, action), self.reward(state))

  def policy(self, state):
    actions = self.available_actions(state)
    if not actions:
      return 0
    q_vals = [(a, self.Q.get((state, a), 0)) for a in actions]
    best_q = max(q_vals, key=lambda x: x[1])[1]
    best_actions = [a for a, q in q_vals if q == best_q]
    return max(best_actions)

  def reward(self, state):
    try:
      c_bag, c_agent, c_opponent, winner = state.split('/')
      c_bag, c_agent, c_opponent = int(c_bag), int(c_agent), int(c_opponent)
    except:
      return 0
    if winner == 'A':
      reward = c_agent
    elif winner == 'O':
      reward = -c_agent
    else:
      reward = 0
    self.rewards[state] = reward
    return reward

  def available_actions(self, state):
    try:
      c_bag = int(state.split('/')[0])
    except:
      return []
    if c_bag == 1:
      return [1]
    elif c_bag == 2:
      return [1, 2]
    elif c_bag >= 3:
      return [1, 2, 3]
    return []


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