import os
import csv
from collections import defaultdict

class td_qlearning:
    eps = 1e-6
    max_iterations = 5000
    alpha = 0.10
    gamma = 0.90

    def __init__(self, directory):
        """
        directory: path to folder containing trial CSV files
        Each CSV file: rows of "state,action"
        """
        self.Q = defaultdict(float)
        self.rewards = {}
        self.trials = []

        # --- Load all trial files ---
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                filepath = os.path.join(directory, file)
                trial_seq = []
                with open(filepath, newline='') as csvfile:
                    reader = csv.reader(csvfile, skipinitialspace=True)
                    for row in reader:
                        if not row or row[0].strip() == "":
                            continue
                        state = row[0].strip()
                        action_str = row[1].strip() if len(row) > 1 else ""
                        # include terminal states (action '-', '', or None)
                        if action_str in ["", "-", "None"]:
                            trial_seq.append((state, None))
                        else:
                            try:
                                trial_seq.append((state, int(action_str)))
                            except ValueError:
                                continue
                if trial_seq:
                    self.trials.append(trial_seq)

        # --- Compute rewards for all states ---
        for trial in self.trials:
            for (state, _) in trial:
                self.rewards[state] = self._reward(state)

        # --- Initialize Q(s,a) = r(s) ---
        for trial in self.trials:
            for (state, action) in trial:
                self.Q[(state, action)] = self.rewards[state]

        # Debug prints
        print("Loaded trials:", len(self.trials))
        if self.trials:
            print("First trial sample:", self.trials[0])

        # --- Train until convergence ---
        self._train()

    # ------------------------------------------------------------------
    def _reward(self, state):
        """Compute reward from state string: cbag/cagent/copponent/winner"""
        parts = state.split('/')
        if len(parts) != 4:
            return 0.0
        bag, agent, opponent, winner = parts
        try:
            agent = int(agent)
        except ValueError:
            return 0.0

        if winner == 'A':
            return float(agent)
        elif winner == 'O':
            return -float(agent)
        else:
            return 0.0

    # ------------------------------------------------------------------
    def _valid_action(self, state, action):
        """Check if action is valid given remaining coins."""
        try:
            bag = int(state.split('/')[0])
        except Exception:
            return False
        return 1 <= action <= 3 and bag >= action

    # ------------------------------------------------------------------
    def _max_q(self, state):
        """Return the maximum Q-value among valid actions for a state."""
        if state is None:
            return 0.0
        actions = [1, 2, 3]
        valid_actions = [a for a in actions if self._valid_action(state, a)]
        if not valid_actions:
            return 0.0
        return max(self.Q[(state, a)] for a in valid_actions)

    # ------------------------------------------------------------------
    def _train(self):
        """Perform temporal difference Q-learning updates."""
        for _ in range(self.max_iterations):
            delta = 0
            for trial in self.trials:
                for i in range(len(trial)):
                    s, a = trial[i]
                    if a is None:
                        continue  # skip terminal

                    # determine next state
                    if i < len(trial) - 1:
                        s_next, _ = trial[i + 1]
                    else:
                        s_next = None

                    # --- reward logic ---
                    # reward is zero for non-terminal next states
                    # reward is ±agent coins for terminal next states
                    if s_next is not None and ('/A' in s_next or '/O' in s_next):
                        r = self.rewards[s_next]   # terminal reward
                        max_q_next = 0.0           # no future value
                    else:
                        r = 0.0
                        max_q_next = self._max_q(s_next) if s_next else 0.0

                    # --- standard TD(0) update ---
                    old_q = self.Q[(s, a)]
                    new_q = old_q + self.alpha * (r + self.gamma * max_q_next - old_q)
                    self.Q[(s, a)] = new_q
                    delta = max(delta, abs(new_q - old_q))
            if delta < self.eps:
                break



    # ------------------------------------------------------------------
    def qvalue(self, state, action):
        """Return learned Q-value for (state, action)."""
        return round(self.Q.get((state, action), 0.0), 2)

    # ------------------------------------------------------------------
    def policy(self, state):
        """Return the optimal action for a given state."""
        actions = [1, 2, 3]
        valid_actions = [a for a in actions if self._valid_action(state, a)]
        if not valid_actions:
            return None
        best_action = max(valid_actions, key=lambda a: self.Q.get((state, a), float('-inf')))
        return best_action


# ------------------------------------------------------------------
# Quick test (safe for submission)
if __name__ == "__main__":
    dir_path = "Examples/Example0/Trials"
    agent = td_qlearning(dir_path)
    print(agent.qvalue("8/3/2/-", 2))   # Expected ≈ 5.67
    print(agent.policy("11/1/1/-"))     # Expected 2
