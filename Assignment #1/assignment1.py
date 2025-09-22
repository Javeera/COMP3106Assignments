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
  # grid contains all the labels from the csv file (X, O, 1-9, S, G)

  #get all the labels (X, O, 1-9, S, G)
  goals = []
  walls = set()
  treasures = {}
  rows = len(grid)
  cols = len(grid[0]) 

  #loop through each coordinate of the grid
  for i in range(rows):
    for j in range(cols):
      if grid[i][j] == 'S':
        start = (i, j)
      elif grid[i][j] == 'G':
        goals.append((i, j))
      elif grid[i][j] == 'X':
        walls.add((i, j))
      elif grid[i][j].isdigit():
        treasures[(i, j)] = int(grid[i][j])

  # optimal_path is a list of coordinate of squares visited (in order)
  # optimal_path_cost is the cost of the optimal path
  # num_states_explored is the number of states explored during A* search
  return optimal_path, optimal_path_cost, num_states_explored
