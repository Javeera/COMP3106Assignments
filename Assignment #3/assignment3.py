# Fatema Lokhandwala (101259465)
# Gurleen
# Javeera

# import statements
import os
import csv
from collections import defaultdict

class td_qlearning:

  alpha = 0.10
  gamma = 0.90

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space
    # Return nothing

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
