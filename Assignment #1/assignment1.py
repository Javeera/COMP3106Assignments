# Gurleen 
# Fatema 
# Javeera Faizi 101191910
# Name this file to assignment1.py when you submit

import csv
import heapq  # For priority queue implementation

# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):
  # filepath is the path to a CSV file containing a grid 
  grid = []
  with open(filepath, newln='') as file:
    reader = csv.reader(file)
    for row in reader:
      grid.append(row)
  return grid

  # optimal_path is a list of coordinate of squares visited (in order)
  # optimal_path_cost is the cost of the optimal path
  # num_states_explored is the number of states explored during A* search
  return optimal_path, optimal_path_cost, num_states_explored
