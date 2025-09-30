# Gurleen Bassali 101260100
# Fatema Lokhandwala 101259465
# Javeera Faizi 101191910
# Name this file to assignment1.py when you submit

import csv
import heapq  # For priority queue implementation

def grid_traversal(filepath):
  # filepath is the path to a CSV file containing a grid 
  grid = []
  with open(filepath, newline='') as file:
    reader = csv.reader(file)
    for row in reader:
      grid.append(row)
  # grid contains all the rows with labels from the csv file (X, O, 1-9, S, G)

  #get all the labels (X, O, 1-9, S, G)
  goals = [] #location of goals
  walls = set() #location of walls
  treasures = {} #stores location & value

  rows = len(grid)
  cols = len(grid[0]) if rows > 0 else 0

  #loop through each coordinate of the grid
  for i in range(rows):
    for j in range(cols):
      if grid[i][j] == 'S':
        start = (i, j)
      elif grid[i][j] == 'G':
        goals.append((i, j))
      elif grid[i][j] == 'X':
        walls.add((i, j))
      elif grid[i][j].isdigit() and int(grid[i][j]) > 0:
        treasures[(i, j)] = int(grid[i][j])

  return start, goals, walls, treasures, rows, cols


def heuristic(a, b):
    #Manhattan distance as heuristic
    #ideal for grids with no diagonals
    #a = [x1, y1], b = [x2, y2]
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):
  start_node, goals, walls, treasures, rows, cols = grid_traversal(filepath)

  # A* search algorithm
  frontier = []
  start_g = 0
  start_h = min(heuristic(start_node, goal) for goal in goals)
  start_f = start_g + start_h
  heapq.heappush(frontier, (start_f, start_g, start_node, 0, set(), [start_node]))  # (f, g, pos, treasure_sum, collected set of treasures, path)

  # best cost so far to reach a specific (node pos, collected_set of treasures) state.
  # Key must be hashable: (pos, tuple(sorted(collected_set)))
  #gScore = { (start_node, ()): 0 } #gScore[state] stores the best cost we have found so far to reach that state.

  num_states_explored = 0
  explored = set() 

  while frontier:
    f, g, current, t_sum, collected, path = heapq.heappop(frontier)

    #key to index into gScore dictionary
    # state_key = (current, tuple(sorted(collected)))
    # if g > gScore[state_key]: # skip exploring node if we’ve already found a cheaper way to this exact state
    #   continue

    #consistent heuristic ensures every time we explore a state (node, set of collected treasures) we've already found optimal cost
    state_key = (current, tuple(sorted(collected)))
    if state_key in explored: #don't re explore if we've already explored this exact state
        continue
    explored.add(state_key)
    num_states_explored += 1

    #check if we are at the goal and have enough treasure
    if current in goals and t_sum >= 5:
      optimal_path = path
      optimal_path_cost = g
      break #return

    # Explore neighbors (up, down, left, right)
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
      neighbour_row, neighbour_col = current[0] + dx, current[1] + dy
      if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols): #if index out of bounds
        continue
      if (neighbour_row, neighbour_col) in walls: #if wall or obstacle
        continue

      #calculate new g, h, f values
      #new_t = t_sum + treasures.get((neighbour_row, neighbour_col), 0)

      # copy the set only if we pick a new treasure
      new_collected = collected
      new_t = t_sum

      # if neighbor is a treasure & we haven't collected it yet
      if (neighbour_row, neighbour_col) in treasures and (neighbour_row, neighbour_col) not in collected:
        new_collected = set(collected)
        new_collected.add((neighbour_row, neighbour_col))
        new_t = t_sum + treasures[(neighbour_row, neighbour_col)]

      new_g = g + 1
      new_h = min(heuristic((neighbour_row, neighbour_col), goal) for goal in goals)
      new_f = new_g + new_h
      #new_key = ((neighbour_row, neighbour_col), tuple(sorted(new_collected)))

      # only push if we found a better path to this exact state (neighbor node, set of collected treasures)
      # if new_g < gScore.get(new_key, float('inf')):
      #   gScore[new_key] = new_g
      heapq.heappush(
        frontier,
        (new_f, new_g, (neighbour_row, neighbour_col), new_t, new_collected, path + [(neighbour_row, neighbour_col)])
      )
      #getting minimum f value
      #heapq.heappush(frontier, (new_f, new_g, (neighbour_row, neighbour_col), new_t, path + [(neighbour_row, neighbour_col)]))

  # optimal_path is a list of coordinate of squares visited (in order)
  # optimal_path_cost is the cost of the optimal path
  # num_states_explored is the number of states explored during A* search
  return optimal_path, optimal_path_cost, num_states_explored

result = pathfinding("Examples\Examples\Example3\grid.txt")
print(result)
